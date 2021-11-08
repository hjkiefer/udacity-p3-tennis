from datetime import datetime
import numpy as np

class StatusPrinter():
    def __init__(self,scores_window_len):
        self.step_size = scores_window_len
        self.time_last_print = datetime.now()
        self.time_last_full_print = datetime.now()
        
    def get_time_string(self, episode):
        now = datetime.now()
        if self._is_new_step(episode):
            t_avg = (now - self.time_last_full_print).total_seconds()/self.step_size
            self.time_last_full_print = now
            self.time_last_print = now
            return f"<t>: {t_avg:2.1f}"
        t = (now-self.time_last_print).total_seconds()
        self.time_last_print = now
        return f" t : {t:2.1f}"
        
    def print_status(self, scores_window, episode):
        ending = self.get_ending(episode)
        print((
            f"\rEpisode {int(episode):3d}"
            f"  Avg: {np.mean(scores_window):4.2f}"
            f"  Cur: {scores_window[-1]:3.3f}"
            f"  Max: {np.amax(scores_window):3.2f}"
            f"  Min: {np.amin(scores_window):3.2f}"
            f"  P25: {np.percentile(scores_window,25):3.2f}"
            f"  P75: {np.percentile(scores_window,75):3.2f}"               
            f"  sum: {np.sum(scores_window):3.2f}"
            #f"  std: {np.std(scores_window):2.3f}"
            f"  {self.get_time_string(episode)}"
            )
            , end=ending)
    
    def _is_new_step(self, episode):
        return episode % self.step_size == 0
    
    def get_ending(self, episode):
        ending = "\n" if self._is_new_step(episode) else ""
        return ending