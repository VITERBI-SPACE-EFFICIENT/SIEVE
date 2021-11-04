#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import re 
import numpy as np 
import random 


def read_triphone_graph(filepath):
    
    '''TRIPHONE GRAPH
   
    Parameters
    filepath: path to triphone file 
    
    Returns 
    map_source_destination :  map from source to destination triphones
    dict_source2labels : map source to labels 
    ''' 
    
    f = open(filepath, "r") 
    allines = f.readlines() 
    map_source_destination = defaultdict(list) 
    dict_source2labels = dict() 
    dict_label2source = dict()
    i = 3 
    
    while i < len(allines): 
        
        row = allines[i]
        splitted = row.split(" ") 
        source_to_name = dict()
        
        if splitted[0] != "": 
            
            source_node = splitted[0][5:-1]
            
            label = splitted[1][:-1]
            
            dict_source2labels[source_node] = label 
                           
            source_to_name[source_node] = splitted[1][:-1] 

            i+=1 
            
            if i < len(allines)-1: 
                
                next_row_splitted = allines[i].split(" ") 
                
                while next_row_splitted[0] == "": 
                    
                    #print( next_row_splitted[-5][1:-1] )
                    for tok in next_row_splitted: 
                        if re.search(r"\[([0-9_]+)\]", tok):
                            break
                       
                    map_source_destination[source_node].append( ( tok[1:-1] , float(next_row_splitted[-1][:-1]) )  )
                    i+=1 
                    next_row_splitted = allines[i].split(" ") 
                    
    
    return map_source_destination , dict_source2labels 
    
     




def read_phone_models(filepath): 
    
    '''
    Params
    filepath: path to input HMM specification file 
    
    Returns 
    
    transition_map: map of transitions 
    allmeans: GMM model means 
    allvariances:GMM model variances 
    allGconsts: GMM model G costants  
    allstates: list of states 
    allprobabilities: list of probabilities 
    alltriphones: list of triphones 
    alltriphonemodels: list of triphone models 
    alltransitions_map:
    triphonemodels_map: 
    
    '''
    
    
    transition_map = dict() 
    f = open(filepath, "r") 

    allines = f.readlines() 
    allines = allines[3:]
    i = 0 
    line = allines[i] 

    splitted = line.split(" ") 
    key = splitted[1][:-1]

    while splitted[0][-1] == "t": 

        i+=1 
        line = allines[i] 

        # next five line are transition matrices 
        A = np.zeros((5,5), dtype = float) 
        for j in range(5): 

            i+=1 
            line = allines[i] 

            this_list = []
            splitted = line.split(" ")[1:] 


            for x in splitted:             
                this_list.append(float(x)) 

            subarray = np.asarray(this_list) 
           # print(subarray.shape)

            subarray.shape = (5,1)

            A[j,:] = this_list 

        transition_map[key] = A 

        i+=1 
        line = allines[i] 
        splitted = line.split(" ") 
        key = splitted[1][:-1]

    i+=2
    line = allines[i] 
    variances_flors = list(map(float, line.split(" ")[1:]))

    i+=1 
    next_line = allines[i].split(" ")

    symbol = next_line[0][-1]
    state = next_line[1][:-1][1:-1]

    allmeans = defaultdict(list) 
    allvariances = defaultdict(list) 
    allGconsts = defaultdict(list) 
    allstates = []
    allstates.append(state)
    allprobabilities = defaultdict(list) 

    while symbol == "s":     

        i+=1 
        line = allines[i].split(" ")

        NUMMIXES = int(line[1])

        for j in range(NUMMIXES): 

            i+=1  
            line = allines[i]
            allprobabilities[state].append(line.split(" ")[-1])


            i+=1  # mean 


            i+=1  
            line = allines[i]


            means = list(map(float, line.split(" ")[1:]))
            allmeans[state].append(means)

            i+=1 

            i+=1 
            line = allines[i]

            variances = list(map(float, line.split(" ")[1:]))
            allvariances[state].append(variances) 

            i+=1 
            line = allines[i]
            G_const = float(line.split(" ")[1])
            allGconsts[state].append(G_const) 


        i+=1 
        line = allines[i]
        next_line = line.split(" ")
        symbol = next_line[0][-1]
        state = next_line[1][:-1][1:-1]
        allstates.append(state) 
        
    allstates = allstates[:-1]    # this is already a triphone model line   

    # when exiting the previous while loop we should have the first h      
    alltriphones = [] 
    alltriphonemodels = []
    alltransitions_map = dict()
    triphonemodels_map = dict() 

    
    while next_line[0][-1] == "h":
        
        st = next_line[1][:-1]
        
        alltriphones.append(st)

        i+=1#1
        line = allines[i]


        i+=1#2
        line = allines[i]

        thistriphone_model = []

        for j in range(3): 

            i+=1#3
            line = allines[i]


            i+=1#4 
            line = allines[i]

            thistriphone_model.append( line.split( " " )[1][:-1] )

        alltriphonemodels.append(thistriphone_model)
        triphonemodels_map[st] = thistriphone_model

        i+=1#5
        line = allines[i]

        if line.split(" ")[0][-1] == "t": 

            alltransitions_map[st] = transition_map[line.split(" ")[1][:-1]]

            i+=1#6 
            line = allines[i]

            i+=1#7 

            if i < len(allines): 

                line = allines[i]

                next_line = line.split(" ")
                
            else: 

                break


        else: 

            i+=1#8
            line = allines[i]


            A = np.zeros((5,5), dtype = float) 
            for j in range(5): 

                this_list = []
                splitted = line.split(" ")[1:] 

              #  print("splitted " + str(splitted))

                for x in splitted:             
                    this_list.append(float(x)) 

                subarray = np.asarray(this_list) 
                # print(subarray.shape)

                subarray.shape = (5,1)

                A[j,:] = this_list 

                i+=1#9 
                line = allines[i]

            alltransitions_map[st] = A


            if i < len(allines): 
                
                i+=1#9 
                line = allines[i]
                
                next_line = line.split(" ")

            else: 

                break
         
    return transition_map, allmeans, allvariances,allGconsts, allstates, allprobabilities , alltriphones, alltriphonemodels, alltransitions_map, triphonemodels_map





def refine_triphone_graph(map_source_destination, dict_source2labels, N): 
    
    ''' random subset of tripohone 
    
    Params: 
    map_source_destination: map from source to destination 
    dict_source2labels:  map from source to labels 
    N: number of nodes     
        
    Returns: 
    new_map_source_destination : refined transitions 
    new_dict_source2labels : refined map from source to labels 
    
    ''' 
    
    new_map_source_destination = dict(random.sample(map_source_destination.items(), N))

    set_keys = set(new_map_source_destination.keys())

    new_dict_source2labels = {k:v for k,v in dict_source2labels.items() if k in set_keys}
        
    return new_map_source_destination, new_dict_source2labels



def refine_triphone_graph_snowball(map_source_destination, dict_source2labels, N): 
    
    ''' snowball sample of triphone graph  
    
    Params: 
    map_source_destination: map from source to destination 
    dict_source2labels:  map from source to labels 
    N: number of nodes     
        
    Returns: 
    new_map_source_destination : refined transitions 
    new_dict_source2labels : refined map from source to labels 
    
    ''' 
    
    first = random.sample(map_source_destination.keys(), 1)[0]
    visited = set() 
    to_visit = []            
    visited.add(first) 
    for ne in map_source_destination[first]: 
        to_visit.append(ne[0]) 
            
    while len(visited) < N: 
        if len(to_visit) == 0: 
            s = random.sample(map_source_destination.keys(), 1)[0]
            while s in visited: 
                s = random.sample(map_source_destination.keys(), 1)[0] 
            to_visit.append(s) 
            
        nod = to_visit.pop(0) 
        visited.add(nod) 
        for neig in map_source_destination[nod]: 
            if neig not in visited: 
                to_visit.append(neig[0])
        
        if len(visited) == N: 
            break 
            
    new_map_source_destination = dict() 
    new_dict_source2labels = dict() 
    for nd in visited: 
        if nd != "23270": 
            new_map_source_destination[nd] = map_source_destination[nd]
            new_dict_source2labels[nd] = dict_source2labels[nd]
    
    # now also updating labels        
    return new_map_source_destination, new_dict_source2labels


def create_complete_network(filepath_grammar, filepath_def, id_start, N): 
    
    ''' assemble entire network from 
    grammar and words 
    
    Params: 
    filepath_grammar : word transitions 
    filepath_def : HMM specification file 
    id_start : start 
    N : number of nodes 
    
    
    Returns: 
    transitions_dict_out: adjancency list of outer transitions 
    transitions_dict_in: adjacency list of inner transtions 
    dict_id2state: map from id to state name 
    all_nodes: list of nodes 
    dict_source2labels: map from source to labels 
    allmeans: list of GMM model means 
    allvariances: list of GMM model variances 
    allstates: list of GMM model states 
    allprobabilities: list of GMM model probabilities 
    '''
    
    transition_map, allmeans, allvariances,allGconsts, allstates, allprobabilities , alltriphones, alltriphonemodels,   alltransitions_map, triphonemodels_map = read_phone_models(filepath_def)
    map_source_destination, dict_source2labels = read_triphone_graph(filepath_grammar)
    transitions_dict_out = defaultdict(list) 
    transitions_dict_in = defaultdict(list) 
    dict_starting = dict() 
    dict_destinations = dict() 
    dict_id2state = dict() 
    all_nodes = list() 
    
    for source in map_source_destination: 
                
        label_key = "\"" + dict_source2labels[source][1:-1] + "\"" 
        label = dict_source2labels[source]
        
        if "{" in label: 
           
            this_triphones_states = [] 
            this_thr_states = triphonemodels_map[label_key]
            
            # add first node 
            id_start+=1 
            this_triphones_states.append(str(id_start) + "s")
            all_nodes.append(str(id_start) + "s")
            dict_destinations[source] = str(id_start) + "s"
                        
            # add middle nodes 
            for i in range(3): 
                id_start+=1 
                this_triphones_states.append( str(id_start) + "s_e" )
                all_nodes.append(  str(id_start) + "s_e" )
                dict_id2state[str(id_start) + "s_e"] = triphonemodels_map[label_key][i] 
                
            # add last node 
            id_start+=1 
            this_triphones_states.append(str(id_start) + "s")
            all_nodes.append(str(id_start) + "s")
            
            # set transitions
            for cnt in range(5):
                st = this_triphones_states[cnt]
                for j in range(5):
                    st_j = this_triphones_states[j]
                    v = alltransitions_map[label_key][cnt][j]
                    if v > 0:
                        transitions_dict_out[st].append( (st_j, np.log(v))  )
                        transitions_dict_in[this_triphones_states[j]].append( (st, np.log(v))  )
                        
            # emission probabilities for the last state 
            last = this_triphones_states[-1]
            dict_starting[source] = last 
                     
        else:
            all_nodes.append(source) 

    # at this point we have replaced triphones and set probabilities within each triphone and among words 
    # now we set the remaining ones (between triphones different and different words) 
    for source in map_source_destination: 
        
        label_source = dict_source2labels[source]
        
        for tup in map_source_destination[source]:
            
            try:
                label_dest = dict_source2labels[tup[0]] # 23270 NOT FOUND: ITS </S> IT SHOULD BE THE END OF THE SENTENCE 
            except:
                continue
        
            if "{" in label_source and "{" not in label_dest: 
                
                transitions_dict_out[dict_starting[source]].append(tup) 
                transitions_dict_in[tup[0]].append( (dict_starting[source], tup[1]))
                
            elif "{" not in label_source and "{" in label_dest: 
                
                transitions_dict_out[source].append( (dict_destinations[ tup[0] ], tup[1]) )
                transitions_dict_in[dict_destinations[ tup[0] ]].append( (source, tup[1]) )
                
            elif "{" not in label_source and "{" not in label_dest:     
                
                
                transitions_dict_out[source].append( tup )
                transitions_dict_in[tup[0]].append( (source, tup[1]) )
                
            else: 
                # both triphones 
                transitions_dict_out[dict_starting[source]].append( (dict_destinations[ tup[0] ], tup[1]) )
                transitions_dict_in[dict_destinations[ tup[0] ]].append( (dict_starting[source], tup[1]) )
    
        
    return transitions_dict_out, transitions_dict_in, dict_id2state, all_nodes , dict_source2labels, allmeans, allvariances, allstates, allprobabilities 





def find_topological_order(first_node, nonemitting, transitions_dict_out): 
    
    ''' find order in which nonemitting states
    
    first_node: initial node (s) 
    nonemitting: set of non emitting states 
    transitions_dict_out: adjacnency list of (out) edges
    
    '''
    
    first = first_node
    to_visit = list() 
    to_visit.extend( list( transitions_dict_out[first] ) )
    
    
    visited = set() 
    visited.add(first)
    
    nonemitting_ordered = []
    nonemitting_ordered.append(first) 
    
    while len(nonemitting.difference(visited)) > 0:
        print(len(nonemitting.difference(visited)))
        nod = to_visit.pop(0)[0]
        for neig in transitions_dict_out[nod]: 
            if neig not in to_visit and neig not in visited: 
                to_visit.append( neig )
       
        nonemitting_ordered.append(nod) 
        visited.add(nod)

    return nonemitting_ordered 