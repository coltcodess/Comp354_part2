import queue
import threading
import time
import random
from train_agent import TrainAgent
from hparams import HParams
import customtkinter as ctk
import gymnasium as gym

import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pygame
from DQNAgent import DQNAgent
import torch

import webbrowser

class TwoStageUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Lunar Lander Trainer")
        self.geometry("1000x700")

        self.progress_queue = queue.Queue()
        self.hyperparam_frame = ctk.CTkFrame(self)
        self.train_progress_frame = ctk.CTkFrame(self)
        self.agent_showcase_frame = ctk.CTkFrame(self)

        # Initialize score tracking and timing
        self.final_score = 0
        self.start_time = None

        # --- Stage 1: Hyperparam selection ---
        self.hyperparam_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(self.hyperparam_frame, text="Set Hyperparameters", font=("Arial", 16)).pack(pady=10)

        # Hyperparameter entries
        ctk.CTkLabel(self.hyperparam_frame, text="Episodes:").pack()
        self.episodes_var = ctk.StringVar(value="10")
        ctk.CTkEntry(self.hyperparam_frame, textvariable=self.episodes_var).pack()

        ctk.CTkLabel(self.hyperparam_frame, text="Learning Rate:").pack()
        self.learning_rate_var = ctk.StringVar(value="0.001")
        ctk.CTkEntry(self.hyperparam_frame, textvariable=self.learning_rate_var).pack()

        ctk.CTkLabel(self.hyperparam_frame, text="Number of Layers:").pack()
        self.n_layers_var = ctk.StringVar(value="2")
        ctk.CTkEntry(self.hyperparam_frame, textvariable=self.n_layers_var).pack()

        ctk.CTkLabel(self.hyperparam_frame, text="Number of Neurons:").pack()
        self.n_neurons_var = ctk.StringVar(value="128")
        ctk.CTkEntry(self.hyperparam_frame, textvariable=self.n_neurons_var).pack()

        ctk.CTkButton(self.hyperparam_frame, text="Start Training", command=self.start_training).pack(pady=20)

        ctk.CTkButton(self.hyperparam_frame, text="Visit Project GitHub", fg_color= "white", hover_color = "grey", text_color = "blue", command=self.open_hyperlink).pack(pady=50)

        self.setup_training_frame()

        # --- Stage 2: After training, show agent frame ---
        ctk.CTkLabel(self.agent_showcase_frame, text="Agent Performance", font=("Arial", 20)).pack(pady=20)
        #self.final_score_label = ctk.CTkLabel(self.agent_showcase_frame, text="Final Score: 0", font=("Arial", 14))
        #self.final_score_label.pack(pady=10)
        # add the agents to test        
        self.trained_agent = None
        #self.trained_agent.load_model()  # Load the trained weights
        
        agents = [self.trained_agent]
        ctk.CTkButton(self.agent_showcase_frame, text="Show Agent in Environment", command=lambda: self.show_agent_performance()).pack(pady=10)
        ctk.CTkButton(self.agent_showcase_frame, text="Train Again", command=self.reset_ui).pack(pady=5)


    def setup_training_frame(self):
        # Training status section
        status_frame = ctk.CTkFrame(self.train_progress_frame)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(status_frame, text="Training Progress", font=("Arial", 16)).pack(pady=5)
        
        self.training_status_label = ctk.CTkLabel(status_frame, text="Training...")
        self.training_status_label.pack(pady=5)
        
        self.elapsed_time_label = ctk.CTkLabel(status_frame, text="Elapsed Time: 0s")
        self.elapsed_time_label.pack(pady=5)

        # Plots section
        plots_frame = ctk.CTkFrame(self.train_progress_frame)
        plots_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(12, 8), tight_layout=True)
        self.ax1 = self.fig.add_subplot(221)  # Episode Rewards
        self.ax2 = self.fig.add_subplot(222)  # Moving Average
        self.ax3 = self.fig.add_subplot(223)  # Episode Length
        self.ax4 = self.fig.add_subplot(224)  # Loss

        # Titles and labels
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')

        self.ax2.set_title('Moving Average (Last 100 episodes)')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Average Reward')

        self.ax3.set_title('Episode Length')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Steps')

        self.ax4.set_title('Training Loss')
        self.ax4.set_xlabel('Episode')
        self.ax4.set_ylabel('Loss')

        # --- Faded (old) lines for each plot (alpha=0.4) ---
        self.old_episode_line, = self.ax1.plot([], [], 'b-', alpha=0.4)
        self.old_movingavg_line, = self.ax2.plot([], [], 'r-', linewidth=2, alpha=0.4)
        self.old_length_line, = self.ax3.plot([], [], 'g-', alpha=0.4)
        self.old_loss_line, = self.ax4.plot([], [], 'm-', alpha=0.4)

        # --- Create persistent Line2D objects for each plot ---
        self.episode_line, = self.ax1.plot([], [], 'b-', alpha=0.7)
        self.movingavg_line, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.length_line, = self.ax3.plot([], [], 'g-', alpha=0.7)
        self.loss_line, = self.ax4.plot([], [], 'm-', alpha=0.7)

        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initialize data storage for plots
        self.episode_rewards = []
        self.moving_averages = []
        self.episode_lengths = []
        self.losses = []
        self.episodes = []

        # --- Add storage for old runs ---
        self.old_episode_rewards = []
        self.old_moving_averages = []
        self.old_episode_lengths = []
        self.old_losses = []
        self.old_episodes = []

    def open_hyperlink(self):
        webbrowser.open("https://github.com/thientran1010/AI_Projects.git")


    def start_training(self):
        try:
            episodes = int(self.episodes_var.get()) if self.episodes_var.get() else 100
            learning_rate = float(self.learning_rate_var.get()) if self.learning_rate_var.get() else 0.001
            n_layers = int(self.n_layers_var.get()) if self.n_layers_var.get() else 2
            n_neurons = int(self.n_neurons_var.get()) if self.n_neurons_var.get() else 128
        except Exception as e:
            print("Invalid hyperparameter input:", e)
            return

        self.hyperparam_frame.pack_forget()
        self.train_progress_frame.pack(fill="both", expand=True)
        
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        
        self.ax2.set_title('Moving Average (Last 100 episodes)')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Average Reward')
        
        self.ax3.set_title('Episode Length')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Steps')
        
        self.ax4.set_title('Training Loss')
        self.ax4.set_xlabel('Episode')
        self.ax4.set_ylabel('Loss')
        
        self.canvas.draw()
        
        self.start_time = time.time()
        self.training_status_label.configure(text="Training...")
        self.elapsed_time_label.configure(text="Elapsed Time: 0s")

        # Start real training thread
        threading.Thread(
            target=self.run_training, 
            args=(episodes, learning_rate, n_neurons, n_layers), 
            daemon=True
        ).start()
        self.after(100, self.poll_progress)

    def run_training(self, total_episodes, learning_rate, n_neurons=128, n_layers=2, num_episodes=50):
        # Create your HParams object
        hparams = HParams(
            n_neurons=n_neurons,
            n_layers=n_layers,
            num_episodes=num_episodes
        )

        # Initialize your TrainAgent with the user-defined hparams
        trainer = TrainAgent(
            model_path="ui_dqn_model.pt",
            num_episodes=total_episodes,
            overwrite=True,
            hparams=hparams,
            learning_rate=learning_rate,
            progress_queue=self.progress_queue  # Pass queue to trainer
        )

        trainer.train_agent()

        # After training is done, send completion signal
        print(trainer, " trainer.")
        print(self.progress_queue, " progress_queue.")
        env = gym.make("LunarLander-v3", render_mode="human")

        self.trained_agent = DQNAgent(env=env, hparams=hparams)
        self.trained_agent.load("ui_dqn_model.pt")  # Load the trained weights
        #final_score = getattr(trainer, 'final_score', 0)  # Get final score if available
        self.progress_queue.put(('done', 0))

    def poll_progress(self):
        try:
            while True:
                result = self.progress_queue.get_nowait()
                if result[0] == 'done':
                    self.final_score = result[1] if len(result) > 1 else 0
                    #self.final_score_label.configure(text=f"Final Score: {self.final_score}")
                    self.training_status_label.configure(text="Training Complete!")
                    self.train_progress_frame.pack_forget()
                    self.agent_showcase_frame.pack(fill="both", expand=True)
                    return
                elif result[0] == 'episode_data':
                    _, episode, reward, avg_reward, episode_length, loss = result
                    self.update_plots(episode, reward, avg_reward, episode_length, loss)

        except queue.Empty:
            pass

        # Update elapsed time
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.elapsed_time_label.configure(text=f"Elapsed Time: {int(elapsed)}s")

        if self.train_progress_frame.winfo_ismapped():
            self.after(100, self.poll_progress)

    def update_plots(self, episode, reward, avg_reward, episode_length, loss):
        self.episodes.append(episode)
        self.episode_rewards.append(reward)
        self.moving_averages.append(avg_reward)
        self.episode_lengths.append(episode_length)
        if loss is not None:
            self.losses.append(loss)

        N = 50  
        self.episode_line.set_data(self.episodes[-N:], self.episode_rewards[-N:])
        self.ax1.relim()
        self.ax1.autoscale_view()

        self.movingavg_line.set_data(self.episodes[-N:], self.moving_averages[-N:])
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.length_line.set_data(self.episodes[-N:], self.episode_lengths[-N:])
        self.ax3.relim()
        self.ax3.autoscale_view()

        loss_episodes = list(range(len(self.losses)))[-N:]
        self.loss_line.set_data(loss_episodes, self.losses[-N:])
        self.ax4.relim()
        self.ax4.autoscale_view()

        self.canvas.draw()



    def run_agent_in_env(self, agent=None, render_mode="human", window_caption=None):
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        obs, _ = env.reset()
        done = False
        total_reward = 0

        # Utility to unwrap to base env (to access .screen)
        def get_base_env(env):
            while hasattr(env, 'env'):
                env = env.env
            return env

        # Patch render to always show score and status
        orig_render = env.render
        def render_with_score():
            result = orig_render()
            pygame.font.init()
            my_font = pygame.font.SysFont('Arial', 30)
            
            # Score text
            text_surf = my_font.render(f"Score: {total_reward:.1f}", True, (255, 255, 255))
            base_env = get_base_env(env)
            base_env.screen.blit(text_surf, (10, 10))
            
            # Success/Fail status
            if total_reward >= 0:
                status_text = "SUCCESS"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "FAIL"
                status_color = (255, 0, 0)  # Red
            
            status_surf = my_font.render(status_text, True, status_color)
            base_env.screen.blit(status_surf, (10, 50))
            
            pygame.display.update()
            return result
        env.render = render_with_score

        if window_caption:
            env.render()
            pygame.display.set_caption(window_caption)

        while not done:
            if agent:
                if not isinstance(obs, torch.Tensor):
                    obs_for_agent = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                else:
                    obs_for_agent = obs
                action = agent.select_action(obs_for_agent)
                if isinstance(action, torch.Tensor):
                    action = action.item()
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            env.render()
            time.sleep(0.05)
        
        env.close()
        
        return total_reward


    def show_agent_performance(self):
        if self.trained_agent is None:
            self.trained_agent = DQNAgent(model_path="ui_dqn_model.pt")

        final_score = self.run_agent_in_env(self.trained_agent, "human", "Trained Agent Demo")

        if final_score >= 200:
            status = "SUCCESS"
            color = "green"
        else:
            status = "FAIL"
            color = "red"

        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title("Final Result")
        popup.geometry("300x150")
    
        result_label = ctk.CTkLabel(popup, text=f"{status}\nScore: {final_score:.2f}", font=("Arial", 18))
        result_label.pack(expand=True, pady=30)

        # Optional: make sure popup stays on top
        popup.lift()
        popup.attributes("-topmost", True)


   

    def reset_ui(self):
        self.agent_showcase_frame.pack_forget()
        self.hyperparam_frame.pack(fill="both", expand=True)
        self.final_score = 0
        #self.final_score_label.configure(text="Final Score: 0")
        self.start_time = None
        
        # --- Save old data for faded lines ---
        self.old_episodes = self.episodes.copy()
        self.old_episode_rewards = self.episode_rewards.copy()
        self.old_moving_averages = self.moving_averages.copy()
        self.old_episode_lengths = self.episode_lengths.copy()
        self.old_losses = self.losses.copy()

        # --- Clear main data for new run ---
        self.episodes.clear()
        self.episode_rewards.clear()
        self.moving_averages.clear()
        self.episode_lengths.clear()
        self.losses.clear()

        # --- Set faded lines to old run, and clear current lines ---
        self.old_episode_line.set_data(self.old_episodes, self.old_episode_rewards)
        self.old_movingavg_line.set_data(self.old_episodes, self.old_moving_averages)
        self.old_length_line.set_data(self.old_episodes, self.old_episode_lengths)
        loss_episodes = list(range(len(self.old_losses)))
        self.old_loss_line.set_data(loss_episodes, self.old_losses)

        self.episode_line.set_data([], [])
        self.movingavg_line.set_data([], [])
        self.length_line.set_data([], [])
        self.loss_line.set_data([], [])

        self.canvas.draw()

if __name__ == "__main__":
    app = TwoStageUI()
    app.mainloop()

