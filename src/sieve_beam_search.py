import numpy as np
from collections import defaultdict
from scipy.stats import multivariate_normal
import copy 
import resource
from math import floor, ceil
import heapq
from operator import itemgetter
from pqdict import PQDict


class SIEVE_BEAMSEARCH:
    
    def __init__(self, pi, A_out, A_in, acustic_costs,  beam_width): 
        self.Pi = pi 
        self.A_out = A_out 
        self.A_in = A_in 
        self.acustic_costs = acustic_costs
        self.initial_state = None 
        self.n_components = 39
        self.visited_median_nodes = set()
        self.B = beam_width  
    
    
    
    def viterbi_space_efficient(self, indices, frames, Pi = None, K = None, last = None, activeTokensStates=None): 
        """
        SIEVE-BS 
        """
        
        T = len(frames) 
                 
        # indices set 
        overall_indices_set = set(copy.deepcopy(indices)) # this indices are the active states 
        
        if K == None: 
            K = len(indices)
        
        if K == 1:           
            print(  [indices[0] for i in range(len(frames))]  ) 
            
  
        if K > 1:     
            Pi = Pi if Pi is not None else defaultdict(lambda: float(1/K))
            T1 = Pi
                        
            previous_n = defaultdict(float)
            previous_medians = defaultdict(lambda: (-1,-1))
            previous_medians_value = defaultdict(lambda: float('inf'))
            previous_active_states = set() 
            
            if activeTokensStates!=None:
                current_indices = activeTokensStates
                
            else:
                current_indices = copy.deepcopy(indices) 
            
            
            for j in frames[1:]: 
                
                new_medians = defaultdict(lambda: (-1,-1))
                new_t1 = defaultdict(lambda: float('-inf'))
                new_n = defaultdict(float)
                new_median_values = defaultdict(lambda: float('inf'))
                updated_medians = set() 
                active_states = defaultdict(set) 
               
                for node_i in current_indices: 
                    
                    for node_h_tuple in self.A_out[ node_i ]: 
                        h = node_h_tuple[0] # index 
                        if h in overall_indices_set and h!=node_i:
                  
                            prob = float( node_h_tuple[1] )
                            if (node_i , h) in self.acustic_costs[j].keys(): 
                                 # transition likelihood 
                                h_mapped_t1 = T1[node_i] + prob + self.acustic_costs[j][(node_i,h)]
                            else:
                                h_mapped_t1 = T1[node_i] + prob
                            
                            
                            if h_mapped_t1  > new_t1[h]: 
                                new_t1[h] = h_mapped_t1 
                               
                                
                                this_pair_to_compare = max(self.b_hop_ancestors[node_i], self.b_hop_descendants[h])
             
                                if this_pair_to_compare < previous_medians_value[node_i] : 
                    
                                    new_median_values[h] = this_pair_to_compare
                                    new_medians[h] = (node_i , h) 
                                    new_n[h] = j 
                                    updated_medians.add(h) 
                                    
                                elif this_pair_to_compare == previous_medians_value[node_i]:
                                    
                                    if abs(j-T/2) < abs(previous_n[node_i]  - T/2): 
                                        
                                        new_median_values[h] = this_pair_to_compare
                                        new_medians[h] = (node_i,h) 
                                        new_n[h] = j 
                                        updated_medians.add(h)
                                        
                                    else:
                                        if previous_medians[node_i]!= (-1,-1): 
                             
                                            new_medians[h] = previous_medians[node_i]
                                            new_n[h] = previous_n[node_i]
                                            new_median_values[h]  = previous_medians_value[node_i] 
                                            if h in updated_medians: 
                                                updated_medians.remove(h) 
                                        
                                            active_states[h] = previous_active_states[node_i] 
                                        
                                else:
                                    if previous_medians[node_i]!=  (-1,-1): 
                                    #    M[i] = previous_medians[maximizer]
                                    #    N[i] = previous_n[maximizer]
                                        new_medians[h] = previous_medians[node_i]
                                        new_n[h] = previous_n[node_i]
                                        new_median_values[h]  = previous_medians_value[node_i] 
                                        if h in updated_medians: 
                                            updated_medians.remove(h) 
                                        active_states[h] = previous_active_states[node_i] 
                                        
                      
                
                effectiveB = min(self.B , len(new_t1))
                current_indices = heapq.nlargest(effectiveB, new_t1, key=new_t1.get)
                
                for nod in updated_medians: 
                    active_states[nod] = current_indices
                
              
                previous_n = new_n
                previous_medians = new_medians 
                previous_medians_value = new_median_values
                T1 = new_t1
                previous_active_states = active_states
                 
                
            if last == None: 
                last = heapq.nlargest(1, T1, key=T1.get)[0] #np.argmax(T1)

            else:
                last = last 
                       
            x_a, x_b =  new_medians[last]   
        
            N_left =  int(new_n[last]) - frames[0]  #floor(len(frames)/2)  
            
            if N_left >1: 
                
                left_frames = frames[:N_left]
                b_hop_ancestors_nodes_x_a = self.single_node_ancestors( x_a , N_left )
                states_left_indices =  sorted( list(b_hop_ancestors_nodes_x_a.union({x_a})) ) # basically indicdes is the list of node string ids (not numeric)
                K_left = len(states_left_indices) # - 1 
                 
                self.viterbi_space_efficient(states_left_indices, left_frames, Pi = Pi, K = K_left, last = x_a, activeTokensStates = activeTokensStates)
                  
                
            N_right = len(frames) - N_left 
            
            print("(" + str(x_a) + " " + str(x_b) + ")")
                                                   
            if N_right >1: 
             
                right_frames = frames[-N_right:]
                b_hop_descendants_nodes_x_b = self.single_node_descendant( x_b , N_right )
                states_right_indices = sorted( list(b_hop_descendants_nodes_x_b.union({x_b})) ) 
                K_right = len(states_right_indices) 
                right_frames = frames[-N_right:]
                pi = defaultdict(lambda: float('-inf'))
                pi[x_b]=0 
                self.viterbi_space_efficient(states_right_indices, right_frames, Pi = pi, K = K_right, last = last, activeTokensStates =  active_states[last])  # append to the right 
                        
                              
        return None
    
    
    
    
    
    def beam_search(self, indices, frames, Pi = None, K = None): 
        """
        STANDARD BEAM SEARCH ALGORITHM 
        """
        
        
        T = len(frames) 
            
       
        tot_memory = len(indices)
        
        if K == None: 
            K = len(indices)
                    
        if self.initial_state!=None: # known initial state 
            #Pi = np.array([-float("inf") if it!=self.initial_state else 0 for it in indices]) # we start from the 
            Pi = defaultdict(lambda: float('-inf'))
            Pi[self.initial_state] = 0                                
            
                                             # initial states for sure   
        Pi = Pi if Pi is not None else defaultdict(lambda: float(1/K))                            
         
        T1 = defaultdict(lambda: defaultdict(lambda: float('-inf')))    
        T2 = defaultdict(lambda: defaultdict(float))    
        
        for t in Pi: 
            T1[0][t] = Pi[t]
            T2[0][t] = 0 
     
        current_indices = copy.deepcopy(indices) 
        
        for j in frames[1:]: 
            
            
           
            this_j_T1 =  defaultdict(lambda: float('-inf')) #[float("-inf")  for _ in range(K)] 
            this_j_T2 =  defaultdict(float) 
        
            for node_i in current_indices: 
             
                for node_h_tuple in self.A_out[ node_i ]: 
                    h = node_h_tuple[0] # index 
                    if h!=node_i:
                    
                        prob = float( node_h_tuple[1] )
                        if (node_i , h) in self.acustic_costs[j].keys(): 
                             # transition likelihood 
                            h_mapped_t1 = T1[j-1][node_i] + prob + self.acustic_costs[j][(node_i,h)]
                        else:
                            h_mapped_t1 = T1[j-1][node_i] + prob
                        
                        
                        if h_mapped_t1  > this_j_T1[h]: 
                            this_j_T1[h] = h_mapped_t1
                            this_j_T2[h] = node_i 
                       
            
            tot_memory+= 2 * len(this_j_T1)
            
            
            for k in this_j_T1:
                T1[j][k] = this_j_T1[k]
                T2[j][k] = this_j_T2[k]               
                     
            
            effectiveB = min(self.B , len(this_j_T1))
            current_indices = heapq.nlargest(effectiveB, this_j_T1, key=this_j_T1.get)
            
        # backtracking 
        x = np.zeros(T, dtype=int)
        
        top_pair = heapq.nlargest(1, T1[T-1], key=T1[T-1].get)
            
        x[-1] = int(top_pair[0])
        top_likelihood =  T1[T-1][top_pair[0]] #float(top_pair[1])  #heapq.nlargest(1, T1[T-1], key=T1[T-1].get)[1]
        
        for i in reversed(range(1, T)):
          
            x[i - 1] = T2[i][x[i]]
                              
        return x , top_likelihood , tot_memory
    
    
    
    
    
    
    
    
    def viterbi_middlepath(self, indices, frames, Pi = None, K = None, last = None, activeTokensStates=None): 
        """
        SIEVE-BS middlepath 
        Space Efficient Beam Search SIEVE-BS-Middlepath algorithm
        """
        
        
        
        th = ceil((frames[0] + frames[-1])/2)
         
        # indices set 
        overall_indices_set = set(copy.deepcopy(indices)) # this indices are the active states 
        
        if K == None: 
            K = len(indices)
     
        
        # Base case number 
        if K == 1:           
            print(  [indices[0] for i in range(len(frames))]  ) 
            
  
        if K > 1:         
            Pi = Pi if Pi is not None else defaultdict(lambda: float(1/K))        
            T1 = Pi                         
            previous_middlepath =  defaultdict(lambda: (-1,-1))
            
            if activeTokensStates!=None:
                current_indices = activeTokensStates
            else:
                current_indices = copy.deepcopy(indices) 
            
            for j in frames[1:]: 
                new_middlepath = defaultdict(lambda: (-1,-1))
                new_t1 = defaultdict(lambda: float('-inf'))
                               
                for node_i in current_indices: 
                    for node_h_tuple in self.A_out[ node_i ]: 
                        h = node_h_tuple[0] # index 
                        if h in overall_indices_set and h!=node_i:
                                
                            prob = float( node_h_tuple[1] )
                            if (node_i , h) in self.acustic_costs[j].keys(): 
                                 # transition likelihood 
                                h_mapped_t1 = T1[node_i] + prob + self.acustic_costs[j][(node_i,h)]
                            else:
                                h_mapped_t1 = T1[node_i] + prob
                            
                            
                            if h_mapped_t1  > new_t1[h]: 
                                new_t1[h] = h_mapped_t1 
                        
                                if  j == th: 
                                    new_middlepath[h] = (node_i , h) 
                                    
                                
                                elif j > th: #floor(T/2): 
                                    new_middlepath[h] = previous_middlepath[node_i]
                                    
                
                
                effectiveB = min(self.B , len(new_t1))
                current_indices = heapq.nlargest(effectiveB, new_t1, key=new_t1.get)

                if j == th: 
                    next_subproblems_indices = current_indices
                              
                previous_middlepath = new_middlepath
                T1 = new_t1
                 
            
            if last == None: 
                last = heapq.nlargest(1, T1, key=T1.get)[0] #np.argmax(T1)

            else:
                last = last 
                       
         
            x_a, x_b =  new_middlepath[last]
            N_left = floor(len(frames)/2)  
            N_right = len(frames) - N_left
            
            if N_left >1: 
                
                left_frames = frames[:N_left]
                b_hop_ancestors_nodes_x_a = self.single_node_ancestors( x_a , N_left )
                states_left_indices =  sorted( list(b_hop_ancestors_nodes_x_a.union({x_a})) ) # basically indicdes is the list of node string ids (not numeric)
                K_left = len(states_left_indices) # - 1 
                self.viterbi_middlepath(states_left_indices, left_frames, Pi = Pi, K = K_left, last = x_a, activeTokensStates = activeTokensStates)
                  
                
            N_right = len(frames) - N_left 
            
            print("(" + str(x_a) + " " + str(x_b) + ")") 
                                                   
            if N_right >1: 
             
                b_hop_descendants_nodes_x_b = self.single_node_descendant( x_b , N_right )
                states_right_indices = sorted( list(b_hop_descendants_nodes_x_b.union({x_b})) ) 
                K_right = len(states_right_indices) 
                right_frames = frames[-N_right:]              
                pi = defaultdict(lambda: float('-inf'))
                pi[x_b]=0 
                
                self.viterbi_middlepath(states_right_indices, right_frames, Pi = pi, K = K_right, last = last, activeTokensStates = next_subproblems_indices)  # append to the right 
                        
        return None
    
    
    

    
    def single_node_descendant(self, source, b): 
        
        visited = set() 
        visited_emitting = dict() 
        
        b_hop_descendants_nodes = set() 
         
        to_be_mantained = set() 
         # Create a queue for BFS
        queue = []
          
        queue.append(source)
        visited_emitting[source] = 1
   
         
        while queue: # and level < b:
        
            s = queue.pop(0)
             
            if visited_emitting[s] < b: 
             
                for tup  in self.A_out[s]:    
                      
                    node_id = tup[0]
                    
                    if node_id not in visited: 
                        b_hop_descendants_nodes.add(node_id) 
                        visited_emitting[node_id] = visited_emitting[s] + 1 
                         
                        queue.append(node_id)
                        visited.add(node_id)
     
        return b_hop_descendants_nodes
        
        
        
    def single_node_ancestors(self,source, b): 

        visited = set() 
        visited_emitting = dict() 
        to_be_mantained = set() 
        b_hop_ancestors_nodes = set() 
         # Create a queue for BFS
        queue = []
        queue.append(source)
        # queue.append("null") # for level 
        visited_emitting[source] = 1

        while queue: # and level < b:

            s = queue.pop(0)
            
             
            if visited_emitting[s] <b : 
             
                for tup  in self.A_in[s]:   
                                          
                    node_id = tup[0]
                    
                    if node_id not in visited: 

                        b_hop_ancestors_nodes.add(node_id) 
                      
                        visited_emitting[node_id] = visited_emitting[s] + 1 
                         
                        queue.append(node_id)
                        visited.add(node_id)    
     
        return b_hop_ancestors_nodes
    
    
