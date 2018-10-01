import tensorflow as tf

def attn_matrix(A, X, attn_weight):
    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F' 
    num_atoms = int(X.get_shape()[1])
    hidden_dim = int(X.get_shape()[2])

    _X1 = tf.einsum('ij,ajk->aik', attn_weight, tf.transpose(X, [0,2,1]))
    _X2 = tf.matmul(X, _X1)
    _A = tf.multiply(A, _X2)
    _A = tf.nn.tanh(_A)

    return _A

def get_skip_connection(_X, X):
    if( int(_X.get_shape()[2]) != int(X.get_shape()[2]) ):
       out_dim = int(_X.get_shape()[2])
       _X = tf.nn.relu(_X + tf.layers.dense(X, units = out_dim, use_bias=False))
    else:
       _X = tf.nn.relu(_X + X) 

    return _X

def get_gate_coeff(X1, X2, dim, label):

    num_atoms = int(X1.get_shape()[1])
    _b = tf.get_variable('mem_coef-'+str(label), initializer=tf.contrib.layers.xavier_initializer(), shape=[dim], dtype=tf.float64)
    _b = tf.reshape(tf.tile(_b, [num_atoms]), [num_atoms, dim])

    X1 = tf.layers.dense(X1, units=dim, use_bias=False)
    X2 = tf.layers.dense(X2, units=dim, use_bias=False)
    
    output = tf.nn.sigmoid(X1+X2+_b)

    return output

def graph_conv_gate(A, X, weight, bias, label):
    dim = int(weight.get_shape()[1])
    num_atoms = int(A.get_shape()[1])

    _b = tf.reshape(tf.tile(bias, [num_atoms]), [num_atoms, dim])
    _X = tf.einsum('ijk,kl->ijl', X, weight) + _b
    _X = tf.matmul(A, _X)
    _X = tf.nn.relu(_X)

    if( int(X.get_shape()[2]) != dim ):
        X = tf.layers.dense(X, dim, use_bias=False)

    coeff = get_gate_coeff(_X, X, dim, label)
    _X = tf.multiply(_X, coeff) + tf.multiply(X,1.0-coeff)

    return _X, coeff

def graph_conv(A, X, weight, bias):
    dim = int(weight.get_shape()[1])
    num_atoms = int(A.get_shape()[1])

    _b = tf.reshape(tf.tile(bias, [num_atoms]), [num_atoms, dim])
    _X = tf.einsum('ijk,kl->ijl', X, weight) + _b
    _X = tf.matmul(A, _X)

    _X = get_skip_connection(_X, X) 

    return _X

def ggnn(A, X, dim, num_layer):
    num_atoms = int(A.get_shape()[1])

    if ( int(X.get_shape()[2]) != dim ):
        X = tf.layers.dense(X, dim, use_bias=False)

    _m = tf.matmul(A, X)

    X_total = []
    #cell = tf.nn.rnn_cell.GRUCell(dim)
    cell = tf.contrib.rnn.GRUCell(dim, name='GRUcell'+str(num_layer))
    for i in range(num_atoms):
        mi = tf.expand_dims(_m[:,i,:],1)
        hi = X[:,i,:]
        #_, _h = tf.nn.static_rnn(cell, mi, initial_state=hi)
        _, _h = tf.nn.dynamic_rnn(cell, mi, initial_state=hi)
        X_total.append(tf.expand_dims(_h,1))
    
    _X = tf.concat(X_total, 1)

    return _X

def graph_attn_gate(A, X, weight, bias, attn, label):
    dim = int(weight[0].get_shape()[1])
    num_atoms = int(A.get_shape()[1])

    X_total = []
    A_total = []
    for i in range( len(weight) ):
        _b = tf.reshape( tf.tile( bias[i], [num_atoms] ), [num_atoms, dim] )
        _h = tf.einsum('ijk,kl->ijl', X, weight[i]) + _b
        _A = attn_matrix(A, _h, attn[i])
        _h = tf.nn.relu(tf.matmul(_A, _h))
        X_total.append(_h)
        A_total.append(_A)

    _X = tf.nn.relu(tf.reduce_mean(X_total, 0))
    _A = tf.reduce_mean(A_total, 0)

    if( int(X.get_shape()[2]) != dim ):
        X = tf.layers.dense(X, dim, use_bias=False)

    coeff = get_gate_coeff(_X, X, dim, label)
    _X = tf.multiply(_X, coeff) + tf.multiply(X,1.0-coeff)

    return _X, _A, coeff

def graph_attn(A, X, weight, bias, attn):
    dim = int(weight[0].get_shape()[1])
    num_atoms = int(A.get_shape()[1])

    X_total = []
    A_total = []
    for i in range( len(weight) ):
        _b = tf.reshape( tf.tile( bias[i], [num_atoms] ), [num_atoms, dim] )
        _h = tf.einsum('ijk,kl->ijl', X, weight[i]) + _b
        _A = attn_matrix(A, _h, attn[i])
        _h = tf.nn.relu(tf.matmul(_A, _h))
        X_total.append(_h)
        A_total.append(_A)

    _X = tf.nn.relu(tf.reduce_mean(X_total, 0))
    _A = tf.reduce_mean(A_total, 0)

    _X = get_skip_connection(_X, X) 

    return _X, _A

def encoder_gat_gate(X, A, num_layers):
    # X : Atomic Feature, A : Adjacency Matrix
    num_atoms = int(X.get_shape()[1])
    input_dim = int(X.get_shape()[2])
    hidden_dim = []
    hidden_dim.append(input_dim)
    for i in range(num_layers):
        hidden_dim.append(32)
    num_attn = 4

    _X = X
    gates = []
    attention = []
    for i in range( len(hidden_dim)-1 ):
        conv_weight = []
        conv_bias = []
        attn_weight = []
        for j in range( num_attn ):
            conv_weight.append( tf.get_variable('ecw'+str(i)+'_'+str(j), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i], hidden_dim[i+1]], dtype=tf.float64) )    
            conv_bias.append( tf.get_variable('ecb'+str(i)+'_'+str(j), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i+1]], dtype=tf.float64) )    
            attn_weight.append( tf.get_variable('eaw'+str(i)+'_'+str(j), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i+1], hidden_dim[i+1]], dtype=tf.float64) )    

        _X, _A, gate = graph_attn_gate(A, _X, conv_weight, conv_bias, attn_weight, i)
        gates.append(gate)
        attention.append(_A)

    return _X, attention, gates

def encoder_gat(X, A, num_layers):
    # X : Atomic Feature, A : Adjacency Matrix
    num_atoms = int(X.get_shape()[1])
    input_dim = int(X.get_shape()[2])
    hidden_dim = []
    hidden_dim.append(input_dim)
    for i in range(num_layers):
        hidden_dim.append(32)
    num_attn = 4

    _X = X
    attention = []
    for i in range( len(hidden_dim)-1 ):
        conv_weight = []
        conv_bias = []
        attn_weight = []
        for j in range( num_attn ):
            conv_weight.append( tf.get_variable('ecw'+str(i)+'_'+str(j), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i], hidden_dim[i+1]], dtype=tf.float64) )    
            conv_bias.append( tf.get_variable('ecb'+str(i)+'_'+str(j), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i+1]], dtype=tf.float64) )    
            attn_weight.append( tf.get_variable('eaw'+str(i)+'_'+str(j), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i+1], hidden_dim[i+1]], dtype=tf.float64) )    

        _X, _A = graph_attn(A, _X, conv_weight, conv_bias, attn_weight)
        attention.append(_A)

    return _X, attention

def encoder_gcn_gate(X, A, num_layers):
    # X : Atomic Feature, A : Adjacency Matrix
    num_atoms = int(X.get_shape()[1])
    input_dim = int(X.get_shape()[2])
    hidden_dim = []
    hidden_dim.append(input_dim)
    for i in range(num_layers):
        hidden_dim.append(32)

    _X = X
    gates = []
    for i in range( len(hidden_dim)-1 ):
        conv_weight = tf.get_variable('ecw'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i], hidden_dim[i+1]], dtype=tf.float64) 
        conv_bias = tf.get_variable('ecb'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i+1]], dtype=tf.float64)

        _X, gate= graph_conv_gate(A, _X, conv_weight, conv_bias, i)
        gates.append(gate)

    return _X, gates

def encoder_gcn(X, A, num_layers):
    # X : Atomic Feature, A : Adjacency Matrix
    num_atoms = int(X.get_shape()[1])
    input_dim = int(X.get_shape()[2])
    hidden_dim = []
    hidden_dim.append(input_dim)
    for i in range(num_layers):
        hidden_dim.append(32)

    _X = X
    for i in range( len(hidden_dim)-1 ):
        conv_weight = tf.get_variable('ecw'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i], hidden_dim[i+1]], dtype=tf.float64) 
        conv_bias = tf.get_variable('ecb'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[i+1]], dtype=tf.float64)

        _X = graph_conv(A, _X, conv_weight, conv_bias)

    return _X

def encoder_ggnn(X, A, num_layers):
    # X : Atomic Feature, A : Adjacency Matrix
    num_atoms = int(X.get_shape()[1])
    input_dim = int(X.get_shape()[2])
    hidden_dim = []
    for i in range(num_layers):
        hidden_dim.append(32)

    _X = X
    for i in range( num_layers ):
        _X = ggnn(A, _X, hidden_dim[i], i)

    return _X

def readout_atomwise(X, latent_size):
    # X : [#Batch, #Atom, #Feature] --> Z : [#Batch, #Atom, #Latent] -- reduce_sum --> [#Batch, #Latent]
    hidden_dim = [latent_size, latent_size, latent_size, 1]
    num_atoms = int(X.get_shape()[1])
    feature_size = int(X.get_shape()[2])
    weight = {
        'mlp_f0': tf.get_variable("fw0", initializer=tf.contrib.layers.xavier_initializer(), shape=[feature_size, hidden_dim[0]], dtype=tf.float64),
        'mlp_f1': tf.get_variable("fw1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
        'mlp_f2': tf.get_variable("fw2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
        'mlp_f3': tf.get_variable("fw3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2], hidden_dim[3]], dtype=tf.float64),
    }
    bias = {
        'mlp_f0': tf.get_variable("fb0", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
        'mlp_f1': tf.get_variable("fb1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
        'mlp_f2': tf.get_variable("fb2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
        'mlp_f3': tf.get_variable("fb3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3]], dtype=tf.float64),
    }
    
    # Graph Embedding in order to satisfy invariance under permutation
    Z = tf.einsum('ijk,kl->ijl', X, weight['mlp_f0']) + tf.reshape(tf.tile(bias['mlp_f0'], [num_atoms]), [num_atoms, hidden_dim[0]])
    Z = tf.nn.relu(Z)
    Z = tf.nn.sigmoid(tf.reduce_sum(Z, 1))

    # Predict the molecular property
    _Y = tf.nn.relu(tf.nn.xw_plus_b(Z, weight['mlp_f1'], bias['mlp_f1']))
    _Y = tf.nn.tanh(tf.nn.xw_plus_b(_Y, weight['mlp_f2'], bias['mlp_f2']))
    _Y = tf.nn.xw_plus_b(_Y, weight['mlp_f3'], bias['mlp_f3'])

    return Z, _Y

def readout_graph_gather(X0, X1, latent_size):
    hidden_dim = [latent_size, latent_size, latent_size, 1]
    num_atoms = int(X0.get_shape()[1])
    feature_size0 = int(X0.get_shape()[2])
    feature_size1 = int(X1.get_shape()[2])
    weight = {
        'mlp_f0i': tf.get_variable("fw0i", initializer=tf.contrib.layers.xavier_initializer(), shape=[feature_size0+feature_size1, hidden_dim[0]], dtype=tf.float64),
        'mlp_f0j': tf.get_variable("fw0j", initializer=tf.contrib.layers.xavier_initializer(), shape=[feature_size1, hidden_dim[0]], dtype=tf.float64),
        'mlp_f1': tf.get_variable("fw1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
        'mlp_f2': tf.get_variable("fw2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
        'mlp_f3': tf.get_variable("fw3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2], hidden_dim[3]], dtype=tf.float64),
    }
    bias = {
        'mlp_f0i': tf.get_variable("fb0i", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
        'mlp_f0j': tf.get_variable("fb0j", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
        'mlp_f1': tf.get_variable("fb1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
        'mlp_f2': tf.get_variable("fb2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
        'mlp_f3': tf.get_variable("fb3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3]], dtype=tf.float64),
    }

    # Graph Embedding in order to satisfy invariance under permutation
    X_concat = tf.concat([X0, X1],2)
    X_i = tf.nn.sigmoid(tf.einsum('ijk,kl->ijl', X_concat, weight['mlp_f0i']) + tf.reshape(tf.tile(bias['mlp_f0i'], [num_atoms]), [num_atoms, hidden_dim[0]]))
    X_j = tf.nn.relu(tf.einsum('ijk,kl->ijl', X1, weight['mlp_f0j']) + tf.reshape(tf.tile(bias['mlp_f0i'], [num_atoms]), [num_atoms, hidden_dim[0]]))
    Z = tf.multiply(X_i, X_j)
    Z = tf.nn.sigmoid(tf.reduce_sum(Z, 1))

    # Predict the molecular property
    _Y = tf.nn.relu(tf.nn.xw_plus_b(Z, weight['mlp_f1'], bias['mlp_f1']))
    _Y = tf.nn.tanh(tf.nn.xw_plus_b(_Y, weight['mlp_f2'], bias['mlp_f2']))
    _Y = tf.nn.xw_plus_b(_Y, weight['mlp_f3'], bias['mlp_f3'])

    return Z, _Y
