import os
import shutil

using_behaviours = True  # behaviours is for gpt data aggregator

if __name__ == "__main__":

    tasks_for_aggregate = ['cover', 'move', 'put_in', 'stack', 'bring', 'uncover', 'push',
                           'press', 'hit', 'rotate', 'close', 'valid']
    aggregate_name = 'gpt_uni'
    base_path = r'D:\datasets\\real_world'

    save_path = os.path.join(base_path, aggregate_name, aggregate_name, 'all_variations', 'episodes')
    os.makedirs(save_path, exist_ok=True)

    merged_episodes = 0
    for task_name in tasks_for_aggregate:

        if using_behaviours:
            task_path = os.path.join(base_path, task_name, 'behaviours', task_name, 'all_variations', 'episodes')
        else:
            task_path = os.path.join(base_path, task_name, task_name, 'all_variations', 'episodes')

        episodes = os.listdir(task_path)
        print('task {} has {} episodes'.format(task_name, len(episodes)))

        for episode in episodes:
            episode_num = int(episode[7:]) + merged_episodes
            new_episode_name = 'episode' + str(episode_num)

            old_episode_path = os.path.join(task_path, episode)
            new_episode_path = os.path.join(save_path, new_episode_name)
            if os.path.exists(new_episode_path):
                print('exist, jump')
                continue
            shutil.copytree(old_episode_path, new_episode_path)

        merged_episodes += len(episodes)
    print('total episodes: {}, train episodes:{}, valid episodes: {}'.format(merged_episodes,
                                                                             merged_episodes - len(episodes),
                                                                             len(episodes)))
