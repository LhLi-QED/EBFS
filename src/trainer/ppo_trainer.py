import os
import ray
import time
import torch
from tqdm import tqdm
from datetime import timedelta

from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from trainer.prompt_dataset import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

from trainer.experience_maker import RemoteProofExperienceMaker, TreeSamplesGenerator

logger = init_logger(__name__)

@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            None,
            None,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            **generate_kwargs,
        )

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        self.samples_generator = TreeSamplesGenerator(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )

        self.experience_maker = RemoteProofExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
        )

        self.prepare_datasets()
        self._init_wandb()

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

        self.rollout_cache = []
        self.all_filtered_ids = []

    def fit(
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)

        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
            )

            for _, rand_prompts, labels in self.prompts_dataloader:
                remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None
                if not len(rand_prompts) % len(self.samples_generator.vllm_engines) == 0:
                    n_rand_prompts = len(rand_prompts) // len(self.samples_generator.vllm_engines) * len(self.samples_generator.vllm_engines)
                    rand_prompts = rand_prompts[:n_rand_prompts]
                    if n_rand_prompts == 0:
                        continue
                else:
                    n_rand_prompts = len(rand_prompts)
                rollout_samples, step_filtered_ids, n_solved = self.samples_generator.generate_samples(
                    rand_prompts, labels, remote_reward_model=remote_reward_model, episode=episode+1, do_eval=False, **self.generate_kwargs
                )
                pbar.update()
                self.all_filtered_ids.extend(step_filtered_ids)
                logger.info(f"Training Solved: {n_solved} / {len(rand_prompts)}")
                
                experiences = self.experience_maker.make_experience_batch(rollout_samples)
                sample0 = self.tokenizer.batch_decode(
                    experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
                )
                experiences = self.rollout_cache + experiences
                m = len(self.actor_model_group._actor_handlers)
                n = len(experiences) // m
                self.rollout_cache = experiences[n*m:]
                experiences = experiences[:n*m]
                print(f"Number of rollout cache: {len(self.rollout_cache)}", sample0)
                refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
                if self.critic_model_group is not None:
                    refs.extend(
                        self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                    )
                ray.get(refs)

                status = self.ppo_train(steps)

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                logger.info(f"✨ Global step {steps}: {status}")
                status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]

                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                steps = steps + 1
        
                if steps % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
                    self.evaluate(self.eval_dataloader, steps, args.eval_temperature, args.eval_n_samples_per_prompt)

            self.prepare_datasets(self.all_filtered_ids)
        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if "generated_samples" in logs_dict:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns, data=self.generated_samples_table.data
                    )
                    new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                    self.generated_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(ref)

    def prepare_datasets(self, filter_out_ids=None):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=self.prompt_split,
        )

        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template, filter_out_ids=filter_out_ids)
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            min(args.vllm_generate_batch_size, len(prompts_dataset)),
            True,
            False,
            drop_last=False
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
                dataset_split=self.eval_split,
            )
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
        else:
            eval_dataloader = None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        if filter_out_ids is None:
            self.max_steps = (
                100 * len(prompts_dataset)
                * args.n_samples_per_prompt
                // args.train_batch_size
                * args.num_episodes
                * args.max_epochs
            )

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources

            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            # Generate samples and calculate rewards
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            _, _, n_solved = self.samples_generator.generate_samples(
                all_prompts, all_labels, remote_reward_model=self.remote_reward_model, episode=1, do_eval=True, **generate_kwargs
            )
            logger.info(f"Evaluation Solved: {n_solved} / {len(all_prompts)}")

            # Calculate global averages
            logs = {f"ratio_solved": n_solved / len(all_prompts)}

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")