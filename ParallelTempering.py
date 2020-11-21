import numpy as np
import threading
import time


#A copy of the Edwards-Anderson Ising model to be run at a specific temperature
class Replica:
    size = 0
    temp = 0.0
    energy = 0
    
    hamiltonian = None
    state = None
    
    min_energy = 0
    min_state = None
    
    def __init__(self, hamiltonian_in, temp_in, state_in = None):
        self.temp = temp_in
        self.hamiltonian = hamiltonian_in
        self.size = hamiltonian_in.shape[0]
        
        if state_in == None:
            A = np.zeros(self.size)
            A.fill(.5)
            rand = np.random.default_rng()
            self.state = 2*(rand.binomial(1, .5, self.size) - A) #initializes nodes to +-1 with probability 1/2
        else:
            self.state = state_in
            
        self.energy = np.matmul(self.state, np.matmul(self.hamiltonian, self.state))
        self.min_energy = self.energy
        self.min_state = self.state
        
    #Monte-Carlo step with single flip update
    def MC_step(self):
        flip = np.random.randint(self.size) #propose a site to flip
        new_state = self.state.copy()
        new_state[flip] = -new_state[flip]
        
        delta_E = np.matmul(new_state, np.matmul(self.hamiltonian, new_state)) - self.energy
        
        if delta_E < 0:
            self.state = new_state
            self.energy += delta_E
        else:
            r = np.random.uniform(0, 1)
            if r < np.exp(-delta_E/self.temp): #rejection sampling
                self.state = new_state
                self.energy+= delta_E
                
        if (self.energy < self.min_energy):
            self.min_energy = self.energy
            self.min_state = self.state

def lowest_energy(ensemble):
    
    energy = 10**6
    state = None
    for config in ensemble:
        if config.min_energy < energy:
            energy = config.min_energy
            state = config.min_state
            
    return energy, state 

def MC_simul(replica, n_steps, signal):
    
    for i in range(n_steps):
        replica.MC_step()
        
    signal["Threads Finished"] += 1

def find_min(hamiltonian_in, temperatures, num_MC_steps, num_swaps):
    start_time = time.time()
    ensemble = [Replica(hamiltonian_in, temp) for temp in temperatures]
    min_energy = 0
    min_energy_state = None
    num_actual_swaps = 0
    
    #run experiment with a designated number of swaps
    for i in range(num_swaps):
    
        min_energy, min_energy_state = lowest_energy(ensemble)
        threads = {}
        signal = {"Threads Finished":0}
    
        for replica in ensemble: #run each replica for N steps in its own thread
            threads[replica] = threading.Thread(target = MC_simul, args = (replica, num_MC_steps, signal))
            threads[replica].start()
        
        while signal["Threads Finished"] != len(ensemble): #wait for threads to finish
            time.sleep(.1) 
    
        min_energy, min_energy_state = lowest_energy(ensemble)
    
        swap = np.random.randint(len(ensemble)-1) #propose to swap
        delta_E = (1/temperatures[swap] - 1/temperatures[swap+1])*(ensemble[swap+1].energy - ensemble[swap].energy)
    
        if delta_E < 0:
            temp_state = ensemble[swap].state.copy()
            ensemble[swap].state = ensemble[swap+1].state.copy()
            ensemble[swap+1].state = temp_state
            num_actual_swaps += 1
        else:
            r = np.random.uniform(0, 1)
            if r < np.exp(-delta_E): #rejection sampling
                temp_state = ensemble[swap].state.copy()
                ensemble[swap].state = ensemble[swap+1].state.copy()
                ensemble[swap+1].state = temp_state
                num_actual_swaps += 1

    return (min_energy_state, min_energy, num_actual_swaps, time.time()-start_time)


def timed_find_min(hamiltonian_in, temperatures, total_time):
    start_time = time.time()
    ensemble = [Replica(hamiltonian_in, temp) for temp in temperatures]
    min_energy = 0
    min_energy_state = None
    num_swaps = 0
    num_actual_swaps = 0
    
    #runs experiment for specified length of time
    while (time.time()-start_time) < total_time:
        num_swaps += 1
        threads = {}
        signal = {"Threads Finished":0}
    
        for replica in ensemble: #run each replica for N steps in its own thread
            threads[replica] = threading.Thread(target = MC_simul, args = (replica, hamiltonian_in.shape[0], signal))
            threads[replica].start()
        
        while signal["Threads Finished"] != len(ensemble): #wait for threads to finish
            time.sleep(.1) 
    
        min_energy, min_energy_state = lowest_energy(ensemble)
    
        swap = np.random.randint(len(ensemble)-1) #propose to swap
        delta_E = (1/temperatures[swap] - 1/temperatures[swap+1])*(ensemble[swap+1].energy - ensemble[swap].energy)
    
        if delta_E < 0:
            num_actual_swaps += 1
            temp_state = ensemble[swap].state.copy()
            ensemble[swap].state = ensemble[swap+1].state.copy()
            ensemble[swap+1].state = temp_state
        else:
            r = np.random.uniform(0, 1)
            if r < np.exp(-delta_E): #rejection sampling
                num_actual_swaps += 1
                temp_state = ensemble[swap].state.copy()
                ensemble[swap].state = ensemble[swap+1].state.copy()
                ensemble[swap+1].state = temp_state
        
    return (min_energy_state, min_energy, num_swaps, num_actual_swaps, time.time()-start_time)

