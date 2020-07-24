import numpy as np
from scipy.spatial import distance
from hopcroftkarp import HopcroftKarp
from sklearn.metrics import pairwise_distances
from rangeFlow import testRangeFlow
from time import time as time

def min_metric(x,X, metric = 'euclidean'):
  distance_matrix = distance.cdist(x, X, metric).flatten()
  return np.min(distance_matrix)


def HeuristicB(X,k, sexes,nr_centers_per_sex,given_centers, metric = 'euclidean'):
    '''Implementation of Heuristic B

    INPUT:
    sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k
    given_centers ... integer-vector with entries in 0,...,n-1

    RETURNS: heuristically chosen centers
    '''
    n = X.shape[0]
    d = X.shape[1]
    m = nr_centers_per_sex.size

    current_nr_per_sex=np.zeros(m)

    if k==0:
        cluster_centers = np.array([], dtype=int)
    else:
        if given_centers.size==0:
            cluster_centers=np.random.choice(n,1,replace=False)
            current_nr_per_sex[sexes[cluster_centers]]+=1
            kk=1
        else:
            cluster_centers = given_centers
            kk=0

        distance_to_closest = np.amin(pairwise_distances(X[cluster_centers].reshape(-1,d), X, metric = metric), axis=0)
        while kk<k:
            feasible_groups=np.where(current_nr_per_sex<nr_centers_per_sex)[0]
            feasible_points=np.where(np.isin(sexes,feasible_groups))[0]
            new_point=feasible_points[np.argmax(distance_to_closest[feasible_points])]
            current_nr_per_sex[sexes[new_point]] += 1
            cluster_centers = np.append(cluster_centers, new_point)
            distance_to_closest = np.amin(np.vstack((distance_to_closest, pairwise_distances(X[new_point].reshape(-1,d), X, metric))), axis=0)
            kk+=1

        cluster_centers=cluster_centers[given_centers.size:]

    return cluster_centers


def gonzalez(X,k, metric = 'euclidean'): 
    '''  
    X :the data set, 2d-aray
    k : is the number of cluster, unsigned int
    RETURNS: list of potential unfair centers
    '''
    C = [] #list of centers to return
    C.append(np.random.randint(0, X.shape[0]))
    K = 1
    kDistance = [] #table storing distance of k centers to other points
    minDist = distance.cdist(X[C], X, metric).flatten()
    kDistance.append(minDist)
    while k!=K:
        candidate = np.argmax(minDist)
        C.append(candidate)
        K = K+1
        newDist = distance.cdist([X[candidate]], X, metric ).flatten()
        kDistance.append(newDist)
        if k!= K:
            minDist = np.min(np.vstack((minDist, newDist)), axis = 0)
    return C, kDistance


def gonzalezNoStore(X,k, metric = 'euclidean'): 
    '''  
    X :the data set, 2d-aray
    k : is the number of cluster, unsigned int
    RETURNS: list of potential unfair centers
    '''
    C = [] #list of centers to return
    C.append(np.random.randint(0, X.shape[0]))
    K = 1
    minDist = distance.cdist(X[C], X, metric).flatten()

    while k!=K:
        candidate = np.argmax(minDist)
        C.append(candidate)
        K = K+1
        newDist = distance.cdist([X[candidate]], X, metric ).flatten()
        if k!= K:
            minDist = np.min(np.vstack((minDist, newDist)), axis = 0)
      
    return C


def gonzalez_variant(X,candidates, k, given_dmat, metric= 'euclidean'): 
    '''  
    X :the data set, 2d-aray
    k : is the number of cluster, unsigned int
    given_dmat: distance matrice that's already computed
    candidates: list of indices considered for greedy selection
    RETURNS: list of potential unfair centers
    '''
    X_sub = X[candidates]
  
    C = [] #list of centers to return
    given_dmat_min = np.min(given_dmat, axis = 0)
    candidate_given_dmat_min = given_dmat_min[candidates]
    C.append(candidates[np.argmax(candidate_given_dmat_min)])
    
    minDist = distance.cdist(X[C], X, metric).flatten()
    candidate_minDist = minDist[candidates]
    if  k == 1: 
        return C,  np.concatenate((given_dmat , [minDist]), axis = 0) 

    K = 1
    kDistance = []  
    candidate_minDist = np.min(np.vstack((candidate_minDist, candidate_given_dmat_min)), axis = 0)
    kDistance.append(minDist)
    while k!=K :
        candidate = np.argmax(candidate_minDist)
        C.append(candidates[candidate])
        K = K+1
        newDist = distance.cdist([X_sub[candidate]], X, metric ).flatten()
        kDistance.append(newDist)
        if k!= K:
            candidate_minDist = np.min(np.vstack((candidate_minDist, newDist[candidates])), axis = 0)

    return C,  np.concatenate((given_dmat , kDistance), axis = 0) 


def testFairShiftFlowFull(centers, closestCenters, distances, classTable, constraints, minDist):
    """
    tests the fair shift using a modified Hopcroft-Karp
    """
    k = len(centers)
    n = len(closestCenters)
    M = len(constraints)
    matchC = [None for i in range(k)] # partners of the centers
    matchG = [[] for i in range(M)]
    num_matched = [0 for i in range(M)]
    match_count = 0
    
    # build the bipartite graph between centers and groups
    neighbors = [[] for i in range(k)]
    for i in range(n):
        if (distances[i] <= minDist):
            neighbors[closestCenters[i]].append((classTable[i], i))

    # initialize greedy matching
    for u in range(k):
        for v, shift_target in neighbors[u]:
            if num_matched[v] < constraints[v]:
                matchC[u] = (v, num_matched[v], shift_target)
                matchG[v].append(u)
                num_matched[v] += 1
                match_count += 1
                break
   
    while True:
        # use BFS to construct layered graph
        distC = [-1 for i in range(k)]
        distG = [-1 for i in range(M)] 
        queue = []
        for u in range(k):
            if matchC[u] == None:
                distC[u] = 0
                queue.append(u)
        head = 0
        found_augmenting_path = False
        while head < len(queue):
            u = queue[head]
            head += 1
            for v, shift_target in neighbors[u]:
                if distG[v] == -1:
                    distG[v] = distC[u] + 1
                    if num_matched[v] < constraints[v]:
                        found_augmenting_path = True
                    for u1 in range(num_matched[v]):
                        distC[matchG[v][u1]] = distG[v] + 1
                        queue.append(matchG[v][u1])

        if not found_augmenting_path:
            if match_count < k:
                return False
            return [x[2] for x in matchC]
            
        # use DFS to find blocking flow in the layered graph
        scan_next = [0 for i in range(M)]
        
        def DFS(u):
            for v, shift_target in neighbors[u]:
                if distG[v] == distC[u] + 1:
                    if num_matched[v] < constraints[v]:
                        matchC[u] = (v, num_matched[v], shift_target)
                        matchG[v].append(u)
                        num_matched[v] += 1
                        return True
                    
                    while scan_next[v] < num_matched[v]:
                        idx = scan_next[v]
                        scan_next[v] += 1
                        u1 = matchG[v][idx]
                        if DFS(u1):
                            matchC[u] = (v, idx, shift_target)
                            matchG[v][idx] = u
                            return True
            return False

        for u in range(k):
            if matchC[u] == None:
                if DFS(u):
                    match_count += 1


def findAllNeighbors(classTable, M, kDistance_i):
  '''
  lassTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k
  M: number of groups
  kDistance_i: vector containing the distance of the center i to all other points
  RETURNS: a list object that closest to center i to each group and a list that contains the corresponding distance
  '''
  
  distTable = [float('infinity')] * M
  neighborTable = [-1] * M
  for D, currentGroup,m in zip(kDistance_i, classTable, list(range(len(classTable)))):
      if D < distTable[currentGroup]:
          distTable[currentGroup] = D
          neighborTable[currentGroup] = m
  return neighborTable, distTable


def HeuristicC(X, classTable, constraints, metric = 'euclidean'):
  """
  Implementation of Heuristic C
  X: dataset
  classTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k 
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  RETURNS: list of fair centers
  """
  k = np.sum(constraints)
  kDistance = []
  M = len(constraints)
  candidate = np.random.randint(len(X))
  fairshift = [candidate]
  k = k - 1
  kDistance = distance.cdist([X[candidate]], X, metric )
  constraints[classTable[candidate]] = constraints[classTable[candidate]]  - 1
  for i in range(M):
      if constraints[i] == 0:
          continue
      else:
          classTableK = [x for x in range(len(classTable)) if classTable[x] == i] 
          candidate = np.array(list(set(classTableK) - set(fairshift)))
          addK, kDistance = gonzalez_variant(X,candidate,constraints[i], kDistance, metric=metric )
          fairshift.extend(addK)
  return fairshift


def recomputeClosestCentersNostore(X, closestCenters, currdist, centers, lorange, hirange, metric = 'euclidean'):
  """
  Update the closest centers of points and range of centers to check, update centers
  X: dataset
  closestCenters: list of closestCenters to update and restore
  currdist: distance between items in X and their closestCenters
  centers: list of centers
  lowrange, hirange: check centers[i] for i in [lorange, hirange)
  """
  # compute distances in batches
  batch = 1000
  index = lorange
  while index < hirange:
    end = min(index + batch, hirange)
    cts = [X[centers[i]] for i in range(index, end)]
    dist = distance.cdist(X, cts, metric)
    closest = np.argmin(dist, axis=1)
    for j in range(X.shape[0]):
      if closestCenters[j] >= hirange - 1 or dist[j][closest[j]] < currdist[j]:
        closestCenters[j] = index + closest[j]
        currdist[j] = dist[j][closest[j]]
    index = end
    
  return closestCenters, currdist




def fairKcenter(X, classTable, constraints, metric='euclidean'):
  """
  Implementation of Alg2-Seq
  X: dataset
  classTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  RETURNS: list of fair centers
  """

  
  k = np.sum(constraints)
  n = X.shape[0]
  classTable = classTable.tolist()
  unfairCenters = gonzalezNoStore(X, k, metric=metric)
  first = 0
  last = k - 1
  fairshift = None
  bestMid = -float('Infinity')
  bestRadius = float('Infinity')
  bestFairShift = None
  
  # instead of remembering distances of classes for each center, instead remember the closest "active" center for each point
  # here we are remembering by index in unfairCenters
  # A is analagous to old_mid
  currentDistanceA = distance.cdist(X, [X[unfairCenters[0]]], metric).flatten()
  currentDistanceMid = distance.cdist(X, [X[unfairCenters[0]]], metric).flatten()
  closestCenterA = [0] * n
  closestCenterMid = [0] * n

  while (first < last):
    mid = (first + last + 1) // 2
    # first, update distances for mid
    closestCenterMid, currentDistanceMid = recomputeClosestCentersNostore(X, closestCenterA.copy(), currentDistanceA.copy(),
                                                                          unfairCenters, first + 1, mid + 1, metric)
    minDist = min_metric([X[unfairCenters[mid]]] , X[unfairCenters[:mid]]) / 2
    if mid > 0:
      start = time()
      fairshift = testFairShiftFlowFull(unfairCenters[:mid + 1], closestCenterMid, currentDistanceMid, classTable,
                                    constraints, minDist)
      print("takes", time() - start)
    else:
      fairshift = True
    if fairshift == False:
      last = mid - 1
    else:
      first = mid
      currentDistanceA = currentDistanceMid
      closestCenterA = closestCenterMid
      bestMid = mid
      bestRadius = minDist
      bestFairShift = np.copy(fairshift)

  mid = bestMid
      
  minDist = bestRadius
  candidateRadius = sorted([x for x in currentDistanceA if x <= minDist])
  
  bestMinDist = minDist

  first = 0
  last = len(candidateRadius) - 1
  fairshift = None

  while (first <= last):
    midRadius = (first + last) // 2
    minDist = candidateRadius[midRadius]
    fairshift = testFairShiftFlowFull(unfairCenters[:mid + 1], closestCenterA, currentDistanceA, classTable, constraints,
                              minDist)
    if fairshift != False and minDist <= bestMinDist:
      bestMinDist = minDist
      bestFairShift = fairshift[:]
      last = midRadius - 1
    else:
      first = midRadius + 1

  classTable = np.array(classTable)
  fairshift = bestFairShift[:]
  constraintsSatisfied, constraintsSatisfiedCount = np.unique(np.array(classTable)[fairshift], return_counts=True)
  for c in range(len(constraintsSatisfied)):
    constraints[constraintsSatisfied[c]] = constraints[constraintsSatisfied[c]] - constraintsSatisfiedCount[c]

  if len(fairshift) == k:
    return fairshift


  for i in range(len(classTable)):
    if i not in fairshift and constraints[classTable[i]] > 0:
      try:
        fairshift.append(i)
      except:
        fairshift = fairshift.tolist()
        fairshift.append(i)
      constraints[classTable[i]] = constraints[classTable[i]] - 1
      if len(fairshift) == k:
        break


  return fairshift

def fairKcenterPlusB(X, classTable, constraints, metric='euclidean'):
  """
  Implementation of Alg2-Seq
  X: dataset
  classTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  RETURNS: list of fair centers
  """
# make the randomness predictable for testing
  
  k = np.sum(constraints)
  n = X.shape[0]
  classTable = classTable.tolist()
  unfairCenters = gonzalezNoStore(X, k, metric=metric)
  first = 0
  last = k - 1
  fairshift = None
  bestMid = -float('Infinity')
  bestRadius = float('Infinity')
  bestFairShift = None
  
  # instead of remembering distances of classes for each center, instead remember the closest "active" center for each point
  # here we are remembering by index in unfairCenters
  # A is analagous to old_mid
  currentDistanceA = distance.cdist(X, [X[unfairCenters[0]]], metric).flatten()
  currentDistanceMid = distance.cdist(X, [X[unfairCenters[0]]], metric).flatten()
  closestCenterA = [0] * n
  closestCenterMid = [0] * n

  while (first < last):
    mid = (first + last + 1) // 2
    # first, update distances for mid
    closestCenterMid, currentDistanceMid = recomputeClosestCentersNostore(X, closestCenterA.copy(), currentDistanceA.copy(),
                                                                          unfairCenters, first + 1, mid + 1, metric)
    minDist = min_metric([X[unfairCenters[mid]]] , X[unfairCenters[:mid]]) / 2
    if mid > 0:
      fairshift = testFairShiftFlowFull(unfairCenters[:mid + 1], closestCenterMid, currentDistanceMid, classTable,
                                    constraints, minDist)
    else:
      fairshift = True
    if fairshift == False:
      last = mid - 1
    else:
      first = mid
      currentDistanceA = currentDistanceMid
      closestCenterA = closestCenterMid
      bestMid = mid
      bestRadius = minDist
      bestFairShift = np.copy(fairshift)

  mid = bestMid
      
  minDist = bestRadius
  candidateRadius = sorted([x for x in currentDistanceA if x <= minDist])
  
  bestMinDist = minDist

  first = 0
  last = len(candidateRadius) - 1
  fairshift = None

  while (first <= last):
    midRadius = (first + last) // 2
    minDist = candidateRadius[midRadius]
    fairshift = testFairShiftFlowFull(unfairCenters[:mid + 1], closestCenterA, currentDistanceA, classTable, constraints,
                              minDist)
    if fairshift != False and minDist <= bestMinDist:
      bestMinDist = minDist
      bestFairShift = fairshift[:]
      last = midRadius - 1
    else:
      first = midRadius + 1

  classTable = np.array(classTable)
  fairshift = bestFairShift[:]
  constraintsSatisfied, constraintsSatisfiedCount = np.unique(np.array(classTable)[fairshift], return_counts=True)
  for c in range(len(constraintsSatisfied)):
    constraints[constraintsSatisfied[c]] = constraints[constraintsSatisfied[c]] - constraintsSatisfiedCount[c]

  if len(fairshift) == k:
    return fairshift

  for i in range(len(classTable)):
    if i not in fairshift and constraints[classTable[i]] > 0:      
      fairshift.append(i)
      constraints[classTable[i]]  =  constraints[classTable[i]]  - 1
      if len(fairshift) == k:
         break
  
  fairshift_ = HeuristicB(X,k-len(fairshift), classTable, np.array(constraints), np.array(fairshift))
  fairshift = np.concatenate((fairshift, fairshift_), axis = None)
  return fairshift


def fairKcenterRange(X, classTable, constraints, lowerbound, k, metric='euclidean'):
  """
  Implementation of Alg2-Seq
  X: dataset
  classTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  RETURNS: list of fair centers
  """
  
  classCount = np.unique(classTable, return_counts=True)[1]
  n = X.shape[0]
  classTable = classTable.tolist()
  unfairCenters = gonzalezNoStore(X, k, metric=metric)
  first = 0
  last = k - 1
  fairshift = None
  bestMid = -float('Infinity')
  bestRadius = float('Infinity')
  bestFairShift = None
  
  # instead of remembering distances of classes for each center, instead remember the closest "active" center for each point
  # here we are remembering by index in unfairCenters
  # A is analagous to old_mid
  currentDistanceA = distance.cdist(X, [X[unfairCenters[0]]], metric).flatten()
  currentDistanceMid = distance.cdist(X, [X[unfairCenters[0]]], metric).flatten()
  closestCenterA = [0] * n
  closestCenterMid = [0] * n

  while (first < last):
    mid = (first + last + 1) // 2
    # first, update distances for mid
    closestCenterMid, currentDistanceMid = recomputeClosestCentersNostore(X, closestCenterA.copy(), currentDistanceA.copy(),
                                                                          unfairCenters, first + 1, mid + 1, metric)
    minDist = min_metric([X[unfairCenters[mid]]] , X[unfairCenters[:mid]]) / 2
    if mid > 0:

      fairshift = testRangeFlow(unfairCenters[:mid + 1], closestCenterMid, currentDistanceMid, classTable,
                                    constraints, lowerbound, classCount, k, minDist)

    else:
      fairshift = True
    if fairshift == False:
      last = mid - 1
    else:
      first = mid
      currentDistanceA = currentDistanceMid
      closestCenterA = closestCenterMid
      bestMid = mid
      bestRadius = minDist
      bestFairShift = np.copy(fairshift)

  mid = bestMid
      
  minDist = bestRadius
  candidateRadius = sorted([x for x in currentDistanceA if x <= minDist])
  
  bestMinDist = minDist

  first = 0
  last = len(candidateRadius) - 1
  fairshift = None

  while (first <= last):
    midRadius = (first + last) // 2
    minDist = candidateRadius[midRadius]
    fairshift = testRangeFlow(unfairCenters[:mid + 1], closestCenterA, currentDistanceA, classTable, constraints,
                                      lowerbound, classCount, k, minDist)
    
    if fairshift != False and minDist <= bestMinDist:
      bestMinDist = minDist
      bestFairShift = fairshift[:]
      last = midRadius - 1
    else:
      first = midRadius + 1

  classTable = np.array(classTable)
  fairshift = bestFairShift[:]
  constraintsSatisfied, constraintsSatisfiedCount = np.unique(np.array(classTable)[fairshift], return_counts=True)
  constraints = lowerbound[:]
  for c in range(len(constraintsSatisfied)):
    constraints[constraintsSatisfied[c]] = constraints[constraintsSatisfied[c]] - constraintsSatisfiedCount[c]

  if len(fairshift) == k:
    return fairshift


  for i in range(len(classTable)):
    if i not in fairshift and constraints[classTable[i]] > 0:
      try:
        fairshift.append(i)
      except:
        fairshift = fairshift.tolist()
        fairshift.append(i)
      constraints[classTable[i]] = constraints[classTable[i]] - 1
      if len(fairshift) == k:
        break


  return fairshift


