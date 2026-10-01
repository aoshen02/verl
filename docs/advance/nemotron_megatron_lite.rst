Nemotron Megatron-lite engine (experimental)
===========================================

Last updated: 10/01/2026.

This fork adds a normal ``EngineRegistry`` backend, ``megatron_lite``, for the
aligned Nemotron-H runtime. It is separate from the external DeepSeek-V4
``verl_mlite`` launchers and from the earlier Transformers/FSDP Nemotron adapter.
Install the matching Megatron-LM runtime and vLLM packages in the same image as
verl; runtime ``PYTHONPATH`` overlays and function replacement installers are
not part of this interface.

Configuration fragment
----------------------

Use the normal engine worker path with an ``ActorConfig`` and explicitly typed
engine configuration. The fragment below is not a validated end-to-end DAPO
recipe. Model and actor worker settings must both retain unpadded inputs.

.. code-block:: yaml

   actor_rollout_ref:
     model:
       use_remove_padding: true
     actor:
       _target_: verl.workers.config.ActorConfig
       strategy: megatron_lite
       engine:
         _target_: verl.workers.engine.megatron_lite.config.MegatronLiteEngineConfig
         strategy: megatron_lite
         dtype: bfloat16
         use_remove_padding: true
         full_determinism: true
         runtime_config:
           model_name: nemotron_h
           parallel:
             tp: 1
             cp: 1
             pp: 1
             vpp: 1

The runtime owns model construction, native backward, optimizer and checkpoint
operations. verl supplies the policy loss and microbatch scheduling. The adapter
preserves packed sample boundaries, restores next-token logprob indexing, and
cancels the runtime's microbatch divisor because verl already normalizes the
policy loss over the global batch. ``BaseEngine.train_batch`` owns the single
optimizer step. Weight export uses the normal checkpoint transport iterator;
layered exports and remote checkpoint retention explicitly fail closed.

Validation boundary
-------------------

The unified-image experiment verified a five-layer functional proxy with the
real hidden/head/expert geometry: 8192 prompt tokens and 1024 response tokens,
1024 exact FP32 logprob matches against an earlier saved rollout, autograd
backward and clean shutdown. This is historical offline forward evidence, not
acceptance of this exact review commit or proof of same-image online RL.
Scalar interface checks verified microbatch normalization and one optimizer
step; they do not establish independent Nemotron gradient accuracy.

Ray TrainingWorker entry, full-model variable-length DAPO, two nonzero policy
updates, resident online weight synchronization and post-update zero-diff are
still required. TP/CP/VPP greater than one, router replay, layered export and
remote checkpoint retention are not supported. Human line-by-line review and
personal test reruns remain required before upstream submission.
