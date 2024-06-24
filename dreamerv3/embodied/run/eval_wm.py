import re

import embodied
import numpy as np
from scipy.stats import zscore


def EvalWM(offline_agent, env, replay, logger, args):

  logdir = embodied.Path(args.logdir)
  offline_directory = args.offline_directory
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', offline_agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')
  # import ipdb; ipdb.set_trace()
  if offline_directory: # eval_pattern=1: offline eval
    driver = embodied.OfflineEvalDriver(env, offline_directory)
  else:
    driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add)

  print('Prefill eval dataset.')
  random_agent = embodied.RandomAgent(env.act_space)
  if not offline_directory:
    while len(replay) < max(args.batch_steps, args.train_fill):
      driver(random_agent.policy, steps=100)
  else:
    while len(replay) < max(args.batch_steps, args.train_fill):
      driver(steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset = offline_agent.dataset(replay.dataset)
  online_state = [None]  # To be writable from train step function below.
  offline_state = [None]
  batch = [None]
  def eval_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
    #   online_outs, online_state[0], mets = agent.train(batch[0], online_state[0])
    #   metrics.add(mets, prefix='eval_online')
      offline_outs, offline_state[0], mets = offline_agent.train(batch[0], offline_state[0])
    #   online_embed = online_outs['embed'].reshape(-1, 4096)
    #   normalized_online_embed = zscore(online_embed, axis=1)
      offline_embeds = []
      for offline_outsi in offline_outs:
        offline_embedi = offline_outsi['embed'].reshape(-1, offline_outsi['embed'].shape[-1])
        offline_embeds.append(offline_embedi)
      offline_embeds = np.array(offline_embeds)
      np.save('/data/ytzheng/code_icml25/dreamerv3_metaworld/scripts/features/offline_embed_offline_input_5emsemble.npy', offline_embeds)
      import ipdb; ipdb.set_trace()
    #   normalized_offline_embed = zscore(offline_embed, axis=1)
    #   embed_mse = np.mean((normalized_offline_embed - normalized_online_embed) ** 2)


      # import ipdb; ipdb.set_trace()
      offline_outs_deter = offline_outs['post']['deter']
      # offline_outs_deter = offline_outs_deter.reshape(-1, offline_outs_deter.shape[-1])
      offline_outs_stoch = offline_outs['post']['stoch']
      offline_outs_stoch = offline_outs_stoch.reshape(offline_outs_stoch.shape[0], offline_outs_stoch.shape[1], -1)
      offline_post = np.concatenate((offline_outs_deter, offline_outs_stoch), -1)
      offline_post = offline_post.reshape(-1, offline_post.shape[-1])

    #   online_outs_deter = online_outs['post']['deter']
    #   # online_outs_deter = online_outs_deter.reshape(-1, online_outs_deter.shape[-1])
    #   online_outs_stoch = online_outs['post']['stoch']
    #   online_outs_stoch = online_outs_stoch.reshape(online_outs_stoch.shape[0], online_outs_stoch.shape[1], -1)
    #   online_post = np.concatenate((online_outs_deter, online_outs_stoch), -1)
    #   online_post = online_post.reshape(-1, online_post.shape[-1])

      # import ipdb; ipdb.set_trace()
    #   if offline_directory:
    #     np.save(f'/data/ytzheng/code_icml25/dreamerv3_metaworld/scripts/features/compare_multiple_features/online_model/offline_input/online_embed{step}_offline_input.npy', online_embed)
    #   else:
    #     np.save(f'/data/ytzheng/code_icml25/dreamerv3_metaworld/scripts/features/compare_multiple_features/online_model/online_input/online_embed{step}_online_input.npy', online_embed)
      # np.save(f'/data/ytzheng/code_icml25/dreamerv3_metaworld/scripts/features/offline_embed021_online_input.npy', offline_embed)
      # np.save(f'/data/ytzheng/code_icml25/dreamerv3_metaworld/scripts/features/offline_embed022_online_input.npy', online_embed)
      # import ipdb; ipdb.set_trace()
      # if 'priority' in outs:
      #   replay.prioritize(outs['key'], outs['priority'])
    #   updates.increment()
    # if should_sync(updates):
    #   agent.sync()
    if should_log(step):
      agg = metrics.result()
    #   online_report = agent.report(batch[0])
    #   online_report = {k: v for k, v in online_report.items() if 'train/' + k not in agg}
      offline_report = offline_agent.report(batch[0])
      offline_report = {k: v for k, v in offline_report.items() if 'train/' + k not in agg}
      logger.add(agg)
    #   logger.add(online_report, prefix='online_report')
      logger.add(offline_report, prefix='offline_report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(eval_step)

  # def eval_offline_step(tran, worker):
  #   for _ in range(should_train(step)):
  #     with timer.scope('dataset'):
  #       batch[0] = next(dataset)
  #     outs, state[0], mets = agent.train(batch[0], state[0])
  #     metrics.add(mets, prefix='eval_offline')
  #     if 'priority' in outs:
  #       replay.prioritize(outs['key'], outs['priority'])
  #     updates.increment()
  #   if should_sync(updates):
  #     agent.sync()
  #   if should_log(step):
  #     agg = metrics.result()
  #     report = agent.report(batch[0])
  #     report = {k: v for k, v in report.items() if 'train/' + k not in agg}
  #     logger.add(agg)
  #     logger.add(report, prefix='report')
  #     logger.add(replay.stats, prefix='replay')
  #     logger.add(timer.stats(), prefix='timer')
  #     logger.write(fps=True)
  # driver.on_step(eval_offline_step)

  # checkpoint = embodied.Checkpoint()
  # # timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  # checkpoint.step = step
  # checkpoint.agent = agent
  # checkpoint.replay = replay
  # # if args.from_checkpoint:
  # #   checkpoint.load(args.from_checkpoint)
  # checkpoint.load_or_save()
  # # should_save(step)  # Register that we jused saved.
#   online_checkpoint = embodied.Checkpoint()
#   online_checkpoint.agent = agent
#   online_checkpoint.load(args.from_online_checkpoint, keys=['agent'])

  offline_checkpoint = embodied.Checkpoint()
  offline_checkpoint.agent = offline_agent
  offline_checkpoint.load(args.from_offline_checkpoint, keys=['agent'])

  print('Start Eval loop.')
  # policy = lambda *args: agent.policy(
  #     *args, mode='explore' if should_expl(step) else 'train')
  if not offline_directory:
    while step < args.steps:
      driver(random_agent.policy, steps=100)
      if should_log(step):
        logger.add(metrics.result())
        logger.add(timer.stats(), prefix='timer')
        logger.write(fps=True)
      # if should_save(step):
      #   checkpoint.save()
  else:
    while step < args.steps:
      driver(steps=100)
      if should_log(step):
        logger.add(metrics.result())
        logger.add(timer.stats(), prefix='timer')
        logger.write(fps=True)
  logger.write()
