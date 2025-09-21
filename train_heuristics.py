from multiprocessing import Process
from env_creator import qsimpy_env_creator
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class HeuristicSolutions:
    def __init__(self, env, num_episodes=100):

        # Initialize the environment
        self.env = env
        self.num_episodes = num_episodes

        # Initialize the results of heuristic solutions
        self.results = []
        # Round Robin index for the QNodes. Example: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...]
        self.rr_index = 0
        # Priority index of Greedy solution after sorting the QNodes based on the waiting time
        self.greedy_index = 0

    def run(self, control):
        """
        Run the heuristic solutions for the given algorithm (control).
        Args:
            - control (str): The heuristic algorithm to use. Options: "greedy", "random", "round_robin", "greedy_error"
        """

        self.results = []
        # Reset the subset of QTasks 
        self.env.unwrapped.round = 1

        # **FIX:** Use an episode counter to provide a unique seed for each episode
        for episode_num in range(self.num_episodes):

            # Initialize the temporary array to store the results of the QTasks execution for each episode
            arr_temp = {
                "total_completion_time": 0.0,
                "rescheduling_count": 0.0
            }
            terminated = False

            # **FIX:** Pass the unique seed here to generate a new problem for each episode
            self.env.reset(seed=episode_num)
            self.env.unwrapped.setup_quantum_resources()
            self.rr_index = 0
            self.greedy_index = 0

            while not terminated:
                # Get the action with the given control
                if control == "greedy":
                    action = self.greedy(self.greedy_index)
                elif control == "random":
                    action = self.random()
                elif control == "round_robin":
                    action = self.round_robin()
                elif control == "greedy_error":
                    action = self.greedy_error(self.greedy_index)
                
                # Save the task object before it gets processed by the step function
                task_to_be_scheduled = self.env.unwrapped.current_qtask
                
                obs, reward, terminated, done, info = self.env.step(action)
                
                # If the action was invalid (e.g., node busy), increment the index for greedy strategies and try again
                if reward < 0: # A reward of -0.1 indicates a penalty for an invalid action
                    self.greedy_index += 1
                
                # If the action was valid and the task was scheduled
                if reward > 0:
                    # Reset priority index of Greedy solution for the next task
                    self.greedy_index = 0

                    # Use the saved task object to access its properties
                    arr_temp["total_completion_time"] += task_to_be_scheduled.waiting_time + task_to_be_scheduled.execution_time
                    arr_temp["rescheduling_count"] += task_to_be_scheduled.rescheduling_count

            sys.stdout.write("\033[F\033[K")
            print(f"[{control}] progress: {episode_num + 1}/{self.num_episodes}")
            self.env.unwrapped.qsp_env.run()
            # Final results of the episode
            self.results.append(arr_temp)

        # Save the results to a CSV file
        self._save_to_csv(control)
                
    def greedy(self, greedy_index):
        # Sort the QNodes based on the next available time (or waiting time) and select the QNode with the smallest waiting time
        greedy_strategy = sorted(self.env.unwrapped.qnodes, key=lambda x: x.next_available_time)
        # Prevent index error if all nodes are invalid
        safe_index = min(greedy_index, len(greedy_strategy) - 1)
        return self.env.unwrapped.qnodes.index(greedy_strategy[safe_index])

    def random(self):
        # Randomly select a QNode
        action = self.env.action_space.sample()
        return action
    
    def round_robin(self):
        # Select the QNode based on the Round Robin index
        action = self.rr_index % self.env.unwrapped.n_qnodes
        self.rr_index += 1
        return action
    
    def greedy_error(self, greedy_index, g_error="Readout_assignment_error"):
        # Sort the QNodes based on the next available time and then by error
        greedy_strategy = sorted(self.env.unwrapped.qnodes, key=lambda x: (x.next_available_time, x.error[g_error]))
        # Prevent index error if all nodes are invalid
        safe_index = min(greedy_index, len(greedy_strategy) - 1)
        return self.env.unwrapped.qnodes.index(greedy_strategy[safe_index])

    def _save_to_csv(self, control) -> None:
        """
        Save values and episodes to a CSV file.
        """

        file_name = "./results/heuristics/" 

        if not os.path.exists(file_name):
            os.makedirs(file_name)

        file_name += control + ".csv"
        # Open the CSV file in write mode
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Episode', 'Total Completion Time', 'Rescheduling Count'])
            
            # Write the data
            for i in range(len(self.results)):
                writer.writerow([i, self.results[i]['total_completion_time'], self.results[i]['rescheduling_count']])
        print("CSV file saved to " + file_name)

    def _plot_results(self, paths) -> None:
        """
        Plot the results of the episodes.
        """
        plt.figure(figsize=(12, 7))
        for path in paths:
            df1 = pd.read_csv(path['path'])

            plt.plot(df1['Episode'], df1['Total Completion Time'], ".-", color=path['color'], label=path['label'])

            self._summarize_results(df1, path['label'])
        
        plt.ylabel('Total Completion Time')
        plt.xlabel('Evaluation Episode')
        plt.title('Heuristic Method Performance Comparison')
        plt.legend(loc='best')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
        plt.show()

    def _summarize_results(self, values, label) -> None:
        """
        Summarize the results of the episodes.
        """
        print("\n--- Results Summary for " + label + " solution ---")
        print(f"Number of Episodes: {self.num_episodes}")
        print(f"Average Total Completion Time: {sum(values['Total Completion Time']) / self.num_episodes:.2f}")
        print(f"Average Rescheduling Count: {sum(values['Rescheduling Count']) / self.num_episodes:.2f}")
        print("------------------------------------------")


if __name__ == "__main__":

    # Create the QSimPy environment
    env_config={
                "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
            }

    env = qsimpy_env_creator(env_config)

    # Run the heuristic solutions
    heuristics = HeuristicSolutions(env, num_episodes=100)
    methods = ['greedy','random','round_robin','greedy_error']
    processes = [Process(target=heuristics.run , args=(m,)) for m in methods]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("All Processes are done!!")

    # Plot the results
    paths = [
        {
            "label": "Random",
            "path": "./results/heuristics/random.csv",
            "color": "red"
        },
        {
            "label": "Round Robin",
            "path": "./results/heuristics/round_robin.csv",
            "color": "blue"
        },
        {
            "label": "Greedy",
            "path": "./results/heuristics/greedy.csv",
            "color": "black"
        },
        {
            "label": "Greedy + Error",
            "path": "./results/heuristics/greedy_error.csv",
            "color": "green"
        },

    ]
    heuristics._plot_results(paths)