import numpy as np

# I assume term_context_matrix
# is in memory now 
# we shall change our implementation
# to loading only term_context_matrix[t]
# to memory later

# you can replace it with your hw4
def create_PPMI_matrix(term_context_matrix, t):
	'''Given a term context matrix, output a PPMI matrix.
	
	See section 15.1 in the textbook.
	
	Hint: Use numpy matrix and vector operations to speed up implementation.
	
	Input:
		term_context_matrix: A nxn numpy array, where n is
				the numer of tokens in the vocab.
	
	Returns: A nxn numpy matrix, where A_ij is equal to the
		 point-wise mutual information between the ith word
		 and the jth word in the term_context_matrix.
	'''       
	
	# YOUR CODE HERE
	term_context_matrix = term_context_matrix[t]
	count_all = np.sum(term_context_matrix)

	def count_w(line_id):
		return np.sum(term_context_matrix, axis = 1)[line_id]

	def count_context(col_id):
		return np.sum(term_context_matrix, axis = 0)[col_id]

	m = np.empty(shape = (len(term_context_matrix), len(term_context_matrix)))
	for i in range(0, len(term_context_matrix)):
		count_word = count_w(i)
		for j in range(0, len(term_context_matrix)):
			pmi = np.log2(term_context_matrix[i,j] * count_all / (count_word * count_context(j)))
			if (pmi > 0).any():
					m[i,j] = pmi
			else: m[i,j] = 0
	return m

# I use BCD here
# may also try SGD later\ 
def train(term_context_matrix, u0, ut, gamma, lam, tau):
	T = term_context_matrix.shape[0]
	U = np.zeros(T + 1)
	U[0] = u0
	U[T] = ut
	for t in range(1, T - 1):
		Yt = term_context_matrix[t]
		# still confused here
		# what to use for Wt
		Wt = np.zeros(d)
		A = np.dot(Wt.T, Wt) + (gamma + lam + 2 * tau) * np.identity(d)
		B = Yt * Wt + gamma * Wt + gamma * (U[t - 1] + U[t + 1])
		U[t] = np.dot(B, np.linalg.inv(A))
	return U










