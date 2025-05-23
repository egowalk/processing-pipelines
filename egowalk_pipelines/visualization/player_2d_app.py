import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from typing import Union
from pathlib import Path
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from egowalk_tools.trajectory import DefaultTrajectory


class TimestampIndexer:

    def __init__(self, timestamps: list[int]):
        self._timestamps = np.array(sorted(timestamps))

    def by_index(self, idx: int) -> int:
        return int(self._timestamps[idx])

    def query(self, timestamp: int) -> tuple[int, int]:
        idx = np.searchsorted(self._timestamps, timestamp, side='right') - 1
        return int(self._timestamps[idx]), idx


class TrajectoryDB2dVisualApp(tk.Tk):
    def __init__(self, 
                 extraction_path: Union[str, Path], 
                 image_backend: str = "opencv"):
        super(TrajectoryDB2dVisualApp, self).__init__()

        trajectory = DefaultTrajectory(extraction_path,
                                       image_backend=image_backend)
        
        self._image_channel = trajectory.rgb_left
        self._image_timestamps = trajectory.rgb_left.timestamps

        self._odometry_traj = trajectory.odometry.to_traj_2d_array()
        self._odom_chunks = self._split_odom(self._odometry_traj)
        self._odometry_indexer = TimestampIndexer(trajectory.odometry.timestamps)

        self._num_frames = len(self._image_timestamps)
        self._current_frame = 0
        self._auto_play = False

        self.title("Trajectory Viewer")
        self.geometry("800x600")

        self._fig, self._ax = plt.subplots()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._img_label = tk.Label(self)
        self._img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._slider = ttk.Scale(self, from_=0, to=self._num_frames - 1, orient="horizontal", command=self._update_frame)
        self._slider.pack(side=tk.BOTTOM, fill=tk.X)

        self._auto_play_button = tk.Button(self, text="Start Auto Play", command=self._toggle_auto_play)
        self._auto_play_button.pack(side=tk.BOTTOM)

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._update_frame(0)

    def _update_frame(self, value):
        frame_idx = int(float(value))
        self._current_frame = frame_idx

        self._display_image(frame_idx)

        self._plot_trajectory(frame_idx)

    def _display_image(self, frame_idx):
        img = self._image_channel[frame_idx]
        img = Image.fromarray(img)
        img = img.resize((400, 300))
        photo = ImageTk.PhotoImage(img)
        self._img_label.config(image=photo)
        self._img_label.image = photo

    def _plot_trajectory(self, frame_idx):
        self._ax.clear()
        for chunk in self._odom_chunks:
            self._ax.plot(chunk[:, 0], chunk[:, 1], 'b-')
        # self._ax.plot(self._odometry_traj[:, 0], self._odometry_traj[:, 1], 'b-')

        image_timestamp = self._image_timestamps[frame_idx]
        odom_timestamp, odom_idx = self._odometry_indexer.query(image_timestamp)
        
        x, y, yaw = self._odometry_traj[odom_idx]
        self._ax.plot(x, y, 'ro')
        self._ax.arrow(x, y, np.cos(yaw), np.sin(yaw), color='r', head_width=0.1)            
        
        self._canvas.draw()

    def _toggle_auto_play(self):
        self._auto_play = not self._auto_play
        if self._auto_play:
            self._auto_play_button.config(text="Stop Auto Play")
            self._auto_play_frames()
        else:
            self._auto_play_button.config(text="Start Auto Play")

    def _auto_play_frames(self):
        if self._auto_play:
            next_frame = (self._current_frame + 1) % self._num_frames
            self._slider.set(next_frame)
            self.after(1, self._auto_play_frames)  # Adjust the interval (in milliseconds) as desired

    def _on_closing(self):
        self._trajectory.close()
        for traj in self._complete_trajectories:
            traj.close()
        self._auto_play = False
        self.destroy()

    def _split_odom(self, odom: np.ndarray) -> list[np.ndarray]:
        chunks = []
        current_chunk = []
        for i, (x, y, yaw) in enumerate(odom):
            if len(current_chunk) > 1:
                point_current = np.array([x, y])
                point_prev = np.array(current_chunk[-1][:2])
                if np.linalg.norm(point_current - point_prev) > 5.0:
                    if np.linalg.norm(point_current) < 1.0:
                        chunks.append(np.array(current_chunk))
                        current_chunk = []
            current_chunk.append((x, y, yaw))
        chunks.append(np.array(current_chunk))
        return chunks
