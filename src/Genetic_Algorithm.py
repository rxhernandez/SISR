import sys
import numpy as np
import itertools

def crossover(parent1, parent2, M0, ul, ll):
    # Combine the parents to create a gene pool
    #N = len(M0)
    gene_pool = np.concatenate((parent1, parent2))
    #print("p1 =", parent1)
    #print("p2 =", parent2)
    #print(gene_pool)
    #print(len(gene_pool))
    #print(np.unique(gene_pool,axis=0))
    has_duplicates = len(np.unique(gene_pool,axis=0)) != len(gene_pool) #remove reactions that are common to both parents
    #print(has_duplicates)
    if has_duplicates == True:
        gene_pool_cut = np.unique(gene_pool,axis=0)
        gene_pool = gene_pool_cut
        #print(len(gene_pool_cut)) 
        total_genes = gene_pool.shape[0]
        max_genes = min(ul, total_genes)        
    
    else:
        total_genes = gene_pool.shape[0]
        max_genes = min(ul, total_genes)

    # Determine a random number of elements for each offspring, capped by max_genes
    num_genes_offspring1 = np.random.randint(ll, max_genes + 1)
    num_genes_offspring2 = np.random.randint(ll, max_genes + 1)
    
    #print(num_genes_offspring1, num_genes_offspring2)
    #sys.exit()
    N = int(len(gene_pool[0])/2)

    # Simple crossover if no M0 is provided
    offspring_work = False
    while offspring_work == False:
        species_included_1 = np.full(N, False)
        species_included_2 = np.full(N, False)
        offspring1_genes_idx = np.random.choice(range(total_genes), size=num_genes_offspring1, replace=False)
        #print(offspring1_genes_idx)
        offspring2_genes_idx = np.random.choice(range(total_genes), size=num_genes_offspring2, replace=False)
        #print(offspring2_genes_idx)
        
        offspring1 = [gene_pool[idx] for idx in offspring1_genes_idx]
        offspring2 = [gene_pool[idx] for idx in offspring2_genes_idx]
        
        for rxn in offspring1:
            species_included_1 = species_in_rxn(species_included_1, rxn, N)
            
        for rxn in offspring2:
            species_included_2 = species_in_rxn(species_included_2, rxn, N)
            
        if np.all(species_included_1) ==True and np.all(species_included_2) == True:
            offspring_work = True
        else:
            offspring_work = False

        
    return offspring1, offspring2

    ## Randomly select the elements for each offspring without replacement
    #offspring = [[], []]
    #for i, num_genes in enumerate([num_genes_offspring1, num_genes_offspring2]):
    #    print("Hello")
    #    sys.exit()
    #    species_included = np.full(N, False)
    #    features_checked = np.full(N, False)
    #    rxn_count = 0
    #    while not (np.all(species_included) and np.all(features_checked) and rxn_count >= ll):
    #        if rxn_count > ul:
    #            rxn_count = 0
    #            offspring[i] = []
    #            species_included = np.full(N, False)
    #            features_checked = np.full(N, False)
    #
    #        random_index = np.random.choice(total_genes)
    #        reaction_i = gene_pool[random_index]
    #        species_included = species_in_rxn(species_included, reaction_i, N)
    #        offspring[i].append(reaction_i)
    #        features_checked = feature_analysis(features_checked,offspring[i],M0)
    #        rxn_count +=1
    #
    #return offspring[0], offspring[1]

# Example function to generate new mechanisms
def generate_new_mechs(mech_ids_sorted, mech_list, prev_gen_mechs, num_new_mechs, M0, m, ul, ll):
    M0 = None
    next_gen_mechs = []
    num_mechs = len(mech_ids_sorted)
	
    num_mechs = len(prev_gen_mechs)

    # Compute selection probabilities based on rank
    probabilities = np.array([(m + (num_mechs - i)) for i in range(num_mechs)])
    probabilities = probabilities / np.sum(probabilities)
	
    probabilities = probabilities[:num_mechs]/ np.sum(probabilities[:num_mechs])
    #print(probabilities)
    #sys.exit()
    # Generate new mechanisms through crossover
    while len(next_gen_mechs) < num_new_mechs:
        # Randomly select two parents using the calculated probabilities
        parents_idx = np.random.choice(range(num_mechs), size=2, replace=False, p=probabilities)

        parent_id1, parent_id2 = mech_ids_sorted[parents_idx[0]], mech_ids_sorted[parents_idx[1]]
        #print(parents_idx,parent_id1, parent_id2)
        parent1 = mech_list[parent_id1[0]]
        parent2 = mech_list[parent_id2[0]]

        offspring1,offspring2 = crossover(parent1, parent2, M0, ul, ll)
        #print("off1 = ", offspring1)
        #print("off2 = ", offspring2)
        #sys.exit()
        is_off1_in_list = mech_check(prev_gen_mechs, offspring1)
        is_off2_in_list = mech_check(prev_gen_mechs, offspring2)
        if is_off1_in_list == False and is_off2_in_list == False: #both offspring are not on mechlist
            next_gen_mechs.extend([offspring1,offspring2])
            #mech_list = mech_list + offspring1 + offspring2 #uncomment if there is a problem generating new mechs
			#print("finished generating", len(next_gen_mechs), "of", num_new_mechs)
        else:
            print("overlap detected in crossover:", "finished generating", len(next_gen_mechs), "of", num_new_mechs)

    # Ensure next_gen_mechs doesn't exceed num_mechs if num_mechs is odd
    next_gen_mechs = next_gen_mechs[:num_new_mechs]
    return next_gen_mechs

def generate_mutations(mech_list, mut_prob, reactions_list):
    # Number of mechanisms and reactions in the mechanism list
    num_mechs = len(mech_list)
    num_reactions = len(reactions_list)
    lambda_mut = mut_prob*num_mechs

    # Generate the number of mutations following a Poisson distribution
    #num_mutations = np.random.poisson(lambda_mut)
    num_mutations = int(lambda_mut)

    # Perform mutations
    for _ in range(num_mutations):
        # Randomly select a mechanism to mutate
        mech_idx = np.random.randint(0, num_mechs)
        selected_mech = mech_list[mech_idx]

        # Randomly select a reaction within the selected mechanism
        reaction_idx = np.random.randint(0, len(selected_mech))
        selected_reaction = selected_mech[reaction_idx]

        # Extract the first 4 elements of the selected reaction This should be the number of species*2
        #reactants = selected_reaction[:4]
        reactants = selected_reaction
        #print("reactants =", reactants, selected_reaction,"old mech =", selected_mech)


        # Find all reactions in the reaction list that have the same first 4 elements (reactants)
        #matching_reactions = np.array([reaction for reaction in reactions_list if np.array_equal(reaction[:4], reactants)])

        # Filter out reactions that are identical to the selected reaction (to ensure mutation)
        #matching_reactions = np.array([reaction for reaction in matching_reactions if not np.array_equal(reaction[4:], selected_reaction[4:])])

        # If there are no valid mutations, skip this iteration
        #if len(matching_reactions) == 0:
        #    continue

        # Randomly select a new reaction from the filtered list of matching reactions
        #new_reaction = matching_reactions[np.random.randint(0, len(matching_reactions))]
        
        number_of_species = int(len(selected_reaction)/2) #takes the full reaction vector and divides by two to get the number of species in the reaction
        species_included = np.full(number_of_species, False) #makes an array of boolean values for that number of species
        
        #Loop over all rxns in the mech except the one that is being replaced to construct species included:
        for rxn in range(len(selected_mech)):
            if rxn == reaction_idx: #this is the rxn your replacing, so don't include in species count
                continue
            else:
                reaction_i = selected_mech[rxn]
                species_included = species_in_rxn(species_included, reaction_i, number_of_species) #last element is number of species
                
        # find a rxn that is not already in the mechanims and that satisfies the species_included constraint 
        is_new_rxn_in_mech = False
        while not (np.all(species_included) and is_new_rxn_in_mech):
            new_reaction_idx = np.random.randint(0, len(reactions_list))
            #print("new rxn ID =", new_reaction_idx)
            new_reaction = reactions_list[new_reaction_idx]
            is_new_rxn_in_mech = array_not_in_list(new_reaction,selected_mech)
            species_included = species_in_rxn(species_included, new_reaction, number_of_species)#last element is number of species
        
        

        # Apply the mutation by replacing the selected reaction in the mechanism with the new reaction
        mech_list[mech_idx][reaction_idx] = new_reaction
        #print("new reaction =", new_reaction)
        #print("new mech =", mech_list[mech_idx])
        #sys.exit()
    return mech_list

def species_in_rxn(species_included, reaction, n_targets):
    """
    Update the list of included species based on whether they appear in a given reaction.

    Parameters:
    - species_included (array-like of bool): Array indicating which species are currently included.
    - reaction (array-like): Array representing a reaction, where the first half corresponds to reactants
      and the second half to products.
    - n_targets (int): Number of target species, used to split the reaction array into reactants and products.

    Returns:
    - array-like of bool: Updated array indicating which species are involved in the reaction.

    Notes:
    - The function marks species as included if they have a non-zero coefficient in either the reactants
      or products of the reaction.
    """

    # Create a copy of the species_included array to update
    species_included_updated = np.copy(species_included)

    # Iterate over each species and check if it is involved in the reaction
    for si in range(n_targets):
        if reaction[si] != 0 or reaction[si + n_targets] != 0:
            species_included_updated[si] = True
    return species_included_updated

def feature_analysis(features_checked,M,M0):
    '''
    Update the list of features that have beenn included based on whether they appear in a given mechanism.

    Parameters:
    - M0 (None or array-like, optional): Initial guesses for what the mechanism looks like constants.
                If array, constraint the mechanism to ensure certain features are included.
    - M0 (None or array-like, optional): Initial guesses for what the mechanism looks like constants.
                If array, constraint the mechanism to ensure certain features are included.

    Returns:
    - array-like of bool: Updated array indicating which features have been included in the mechanism thus far.
    '''

    features_checked_updated = np.copy(features_checked)
    N = len(M0)

    for i, trend in enumerate(M0):
        coeff_ids = np.where(trend > 0)[0]
        if len(coeff_ids) == 0:
            features_checked_updated[i] = True
            break
        if len(coeff_ids) == 1:
            for rj, reaction in enumerate(M):
                active_features = trend*reaction
                constraint = active_features[N:]-active_features[:N]
                #print(trend,reaction,active_features,constraint)
                if not np.all(constraint==0):
                    features_checked_updated[i] = True
                    break
        elif len(coeff_ids) == 2:
            trend1 = trend[:N]
            trend2 = trend[N:]
            for rj, reaction in enumerate(M):
                active_features1 = trend1*reaction[:N]
                active_features2 = trend2*reaction[N:]
                constraint = active_features2-active_features1
                #print(trend,reaction,active_features1,active_features2,constraint)
                if not np.all(constraint==0):
                    features_checked_updated[i] = True
                    break
        else:
            print(M0)
            raise ValueError(f'Invalid number of elements in vector M0, index = {i}')
    return features_checked_updated
	
def mech_check(mechanism_list, mech_i):
  """Checks if mech is already present in a list of arrays."""
  #print("mech check for =", mech_i)
  #print("mech_list =", mechanism_list)
  are_equal = False
  mech_is_in_mech_list = False
  for other_mech in mechanism_list:
    permutations = list(itertools.permutations(other_mech, len(other_mech)))
    for other_mech_permutation in permutations:
      are_equal = all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(mech_i, other_mech_permutation))
      if are_equal == True:
        #print(are_equal, mech_i, other_mech, other_mech_permutation)
        #print("permutations =", permutations)
        mech_is_in_mech_list = True
        continue #sys.exit()
  #if are_equal == True:
  #if is_in_list == True:
    #return True
  #else:
  return mech_is_in_mech_list 



def array_not_in_list(arr, list_of_arrays):
  """Checks if an array is not present in a list of arrays."""

  for other_arr in list_of_arrays:
      if (arr == other_arr).all():
          return False
  return True
  
def remove_one_duplicate(arr):
    """Removes one instance of each duplicate element from a NumPy array."""

    unique, indices = np.unique(arr, return_index=True)
    return arr[np.sort(indices)]


# Example usage
#mechanism_list = [...]  # List of current mechanisms with their fitness errors
#num_mechs = self.num_mechs
#prev_gen_count = self.prev_gen
#crossover_prob = 0.8
#new_mechs = generate_new_mechs(mechanism_list, num_mechs, prev_gen_count, crossover_prob)
