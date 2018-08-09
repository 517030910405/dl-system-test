import numpy as np

def get_pointer(input):
    return input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
global_variables = {}
float32 = np.float32
float64 = np.float64
float16 = np.float16
float_ = np.float_
int64 = np.int64
int32 = np.int32
int16 = np.int16
int8 = np.int8
int_ = np.int_
bool_ = np.bool_
zeros = np.zeros
def simple_list(list_A):
    ans=[]
    for i in list_A:
        if isinstance(i,train_type):
            ans += simple_list(i.data)
        elif isinstance(i,list):
            ans += simple_list(i)
        elif True:
            ans. append(i)
    return ans

class global_variables_initializer_type:
    def __init__(self):
        assert True
def global_variables_initializer():
    return global_variables_initializer_type()
class train_type(object):
    def __init__(self, data = [], info = None):
        self.data = data
        #self.vari = []
    def run(self, feed_dict={}):
        sess = Session()
        sess.run([self], feed_dict = feed_dict )
class assign_type(object):
    def __init__(self,node,val):
        self.node = node
        self.val = val
def assign(node, val):
    return assign_type(node, val)
def random_normal(shape= None, mean = 0.0, stddev = 1.0, dtype = float32, name = "random_normal"):
    return np.random.normal(size = tuple(shape), loc = mean, scale = stddev)
def constant(value = 0, dtype = float64, shape = None, name = "Const"):
    value = np.array(value).astype(dtype)
    if shape != None:
        return np.broadcast_to(np.array(value), tuple(shape))
    else:
        return value
class train:
    class GradientDescentOptimizer:
        def __init__(self, learning_rate):
            self. learning_rate = learning_rate
        def minimize(self, eval_node):
            variables_list = []
            topo_order = find_topo_sort([eval_node])
            for node in topo_order:
                if node in global_variables:
                    variables_list.append(node)
            ans = []
            grad_list = gradients(eval_node, variables_list)
            for i in range(len(variables_list)):
                ans.append(assign(variables_list[i], variables_list[i] - grad_list[i] * self.learning_rate) )
            return train_type(ans)
    class AdamOptimizer:
        def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, name = "Adam"):
            self. learning_rate = float64(learning_rate)
            self. beta1 = float64(beta1)
            self. beta2 = float64(beta2)
            self. epsilon = float64(epsilon)
            self. name = name
        def minimize(self, eval_node):
            variables_list = []
            topo_order = find_topo_sort([eval_node])
            for node in topo_order:
                if node in global_variables:
                    variables_list.append(node)
            ans = []
            t = Variable(1,dtype = float64, name = "t")
            grad_list = gradients(eval_node, variables_list)
            ans.append(assign(t, t + 1 ) )
            fr1 = (1-exp(np.log((self.beta1))*t)) 
            fr2 = (1-exp(np.log((self.beta2))*t)) 
            lr = self.learning_rate * sqrt_op( fr2/fr1 )
            lr.name="lr"
            for i in range(len(variables_list)):
                m = Variable(np.zeros(global_variables[variables_list[i]].shape), name = "mt")
                new_m = self.beta1*m+(1-self.beta1)*grad_list[i]
                ans.append(assign(m,new_m))
                v = Variable(np.zeros(global_variables[variables_list[i]].shape), name = "vt")
                new_v = self.beta2*v+(1-self.beta2)*grad_list[i]*grad_list[i]
                ans.append(assign(v,new_v))
                delta = lr*new_m/(sqrt_op(new_v)+self.epsilon)
                ans.append(assign(variables_list[i], variables_list[i] - delta))
            return train_type(data = ans)
        
class Node(object):
    def eval(self, feed_dict ={}):
        sess = Session()
        return sess.run(self, feed_dict)
    def __init__(self):
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.dtype = None
    def __add__(self, other):
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            new_node = add_byconst_op(self, other)
        return new_node
    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            new_node = sub_byconst_op(self, other)
        return new_node
    def __rsub__(self, other):
        new_node = rsub_byconst_op(self, other)
        return new_node
    def __neg__(self):
        return 0-self
    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node =mul_byconst_op(self, other)
        return new_node
    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_byconst_op(self, other)
        return new_node
    def __rtruediv__(self, other):
        new_node = rdiv_byconst_op(self, other)
        return new_node
    # Allow left-hand-side
    __radd__ = __add__
    __rmul__ = __mul__
    def __str__(self):
        return self.name
    __repr__ = __str__
def argmax(node_A, axis = None):
    return argmax_op(node_A, axis)
def matmul(node_A, node_B):
    return matmul_op(node_A, node_B)
def reduce_sum(node, reduction_indices=None):
    new_node = reduce_sum_op(node , reduction_indices)
    return new_node
def broadcast_to(node_A, node_B, reduction_indices=None):
    new_node = broadcast_to_op(node_A, node_B , reduction_indices)
    return new_node
def reduce_mean(node, reduction_indices=None):
    new_node = reduce_mean_op(node , reduction_indices)
    return new_node
def broadcast_mean_to(node_A, node_B, reduction_indices=None):
    new_node = broadcast_mean_to_op(node_A, node_B , reduction_indices)
    return new_node
def cast(node_A, dtype=float64):
    new_node = cast_op(node_A, dtype)
    return new_node
def equal(node_A, node_B):
    new_node = equal_op(node_A, node_B)
    return new_node
def exp(node_A):
    if isinstance(node_A, Node):
        new_node = exp_op(node_A)
        return new_node
    else:
        return np.exp(node_A)
def log(node_A):
    if isinstance(node_A, Node):
        new_node = log_op(node_A)
        return new_node
    else:
        return np.log(node_A)
def Variable(val, dtype = None, name="Variable", shape = None):
    placeholder_node = placeholder_op()
    if dtype!=None:
        global_variables[placeholder_node] = np.array(val).astype(dtype)
    else:
        global_variables[placeholder_node] = np.array(val)
    return placeholder_node
def placeholder(dtype=float64, name="holder",shape = None):
    placeholder_node = placeholder_op()
    placeholder_node.dtype = dtype
    return placeholder_node
class Op(object):
    def __call__(self):
        """Create a new node and associate the op object with the node.
        Returns
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node
    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.
        Parameters
        node: node that performs the compute.
        input_vals: values of input nodes.
        Returns
        An output value of the node.
        """
        raise NotImplementedError
    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.
        Parameters
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions
        Returns
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError
class GetShapeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.array(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]
#edition 11 get_shape_op
class ReshapeConstOp(Op):
    def __call__(self, node_A, shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = tuple(shape)
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0].reshape(tuple(node.const_attr))
    def gradient(self, node, output_grad):
        return [reshape_op(output_grad , get_shape_op(node.inputs[0]))]
class ReshapeOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0].reshape(tuple(input_vals[1]) )
    def gradient(self, node, output_grad):
        return [reshape_op(output_grad , get_shape_op(node.inputs[0])), zeroslike_op(node.inputs[1])]
#edition 11 reshape_op
class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] + input_vals[1]
    def gradient(self, node, output_grad):
        return [auto_sum_op(output_grad, get_shape_op(node.inputs[0]) ), auto_sum_op(output_grad, get_shape_op(node.inputs[1]) )]
class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] + node.const_attr
    def gradient(self, node, output_grad):
        return [output_grad]
class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] - input_vals[1]
    def gradient(self, node, output_grad):
        return [auto_sum_op(output_grad, get_shape_op(node.inputs[0]) ), 0-auto_sum_op(output_grad, get_shape_op(node.inputs[1]) )]
class SubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0] - node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]
class RSubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node
    def compute(self, node, input_vals):
        return node.const_attr - input_vals[0] 
    def gradient(self, node, output_grad):
        return [ - output_grad]
class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] * input_vals[1]
    def gradient(self, node, output_grad):
        return [auto_sum_op(output_grad * node.inputs[1] , get_shape_op(node.inputs[0])), auto_sum_op(output_grad * node.inputs[0] , get_shape_op(node.inputs[1]))]
class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] * node.const_attr
    def gradient(self, node, output_grad):
        return [output_grad * node.const_attr ]
class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        if node.matmul_attr_trans_A :
            input_vals[0] = input_vals[0].T
        if node.matmul_attr_trans_B :
            input_vals[1] = input_vals[1].T
        return np.matmul(input_vals[0] , input_vals[1])
    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        hahaha233 = MatMulOp()
        return [ hahaha233( output_grad, node.inputs[1], False , True) , hahaha233( node.inputs[0] , output_grad , True , False ) ]
class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [auto_sum_op(output_grad / node.inputs[1] ,get_shape_op(node.inputs[0])), auto_sum_op(-output_grad * node.inputs[0] / node.inputs[1] / node.inputs[1] , get_shape_op(node.inputs[1]) ) ]
class DivByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0] / node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad / node.const_attr ]
class RDivByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return node.const_attr / input_vals[0] 

    def gradient(self, node, output_grad):
        return [ - output_grad * node.const_attr / node.inputs[0] / node.inputs[0]  ]
class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp(node.inputs[0])]
class LogOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [ output_grad / node.inputs[0] ]
class ReduceSumOp(Op):
    def __call__(self, node_A, reduction_indices=None):
        new_node = Op.__call__(self)
        if reduction_indices==None:
            new_node.const_attr = None
        else:
            reduction_indices.sort()
            new_node.const_attr = tuple(reduction_indices)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        if node.const_attr!=None:
            return np.array(np.sum(input_vals[0], node.const_attr))
        else:
            return np.array(np.sum(input_vals[0]))

    def gradient(self, node, output_grad):
        return [broadcast_to(output_grad,get_shape_op(node.inputs[0]),node.const_attr)]
class ReduceMeanOp(Op):
    def __call__(self, node_A, reduction_indices=None):
        new_node = Op.__call__(self)
        if reduction_indices==None:
            new_node.const_attr = None
        else:
            reduction_indices.sort()
            new_node.const_attr = tuple(reduction_indices)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        if node.const_attr!=None:
            return np.array(np.mean(input_vals[0], node.const_attr))
        else:
            return np.array(np.mean(input_vals[0]))

    def gradient(self, node, output_grad):
        return [broadcast_mean_to(output_grad,get_shape_op(node.inputs[0]),node.const_attr)]

""" edition 12: change to getshape style """
class BroadcastToOp(Op):
    """using the broadcast system"""
    def __call__(self, node_A, node_B, reduction_indices=None):
        new_node = Op.__call__(self)
        if reduction_indices==None:
            new_node.const_attr = None
        else:
            new_node.const_attr = tuple(reduction_indices)
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        if node.const_attr!=None:
            shape = tuple(input_vals[1])
            oldshape = list(input_vals[0].shape)
            for i in node.const_attr:
                oldshape.insert(i%(len(oldshape)+1),1)
            return np.array(np.broadcast_to(input_vals[0].reshape(tuple(oldshape)),shape))
        else:
            return np.broadcast_to(input_vals[0], tuple(input_vals[1]))

    def gradient(self, node, output_grad):
        if node.const_attr==None:
            return [reduce_sum(output_grad,reduction_indices = None),zeroslike_op(node.inputs[1])]
        else:
            return [reduce_sum(output_grad,reduction_indices = list(node.const_attr)),zeroslike_op(node.inputs[1])]

class BroadcastMeanToOp(Op):
    def __call__(self, node_A, node_B, reduction_indices=None):
        new_node = Op.__call__(self)
        if reduction_indices==None:
            new_node.const_attr = None
        else:
            new_node.const_attr = tuple(reduction_indices)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        shape = tuple(input_vals[1])
        divby = 1
        if node.const_attr!=None:
            oldshape = list(input_vals[0].shape)
            for i in node.const_attr:
                oldshape.insert(i%(len(oldshape)+1),1)
                divby *= shape[i]
            return np.array(np.broadcast_to(input_vals[0].reshape(tuple(oldshape)),shape))/divby
        else:
            for i in shape:
                divby *= i
            return np.broadcast_to(input_vals[0], shape)/divby
    def gradient(self, node, output_grad):
        if node.const_attr==None:
            return [reduce_mean(output_grad,reduction_indices = None),zeroslike_op(node.inputs[1])]
        else:
            return [reduce_mean(output_grad,reduction_indices = list(node.const_attr)),zeroslike_op(node.inputs[1])]
class AutoBroadcastOp(Op):
    """using the broadcast system"""
    def __call__(self, node_A, node_B):
        #node_B is the other shape 
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        shape_A = list(input_vals[0].shape)
        shape_B = list(input_vals[1])
        while len(shape_A)<len(shape_B):
            shape_A.insert(0,1)
        while len(shape_B)<len(shape_A):
            shape_B.insert(0,1)
        new_shape = []
        for i in range(len(shape_A)):
            new_shape.append(max(shape_A[i],shape_B[i]))
        ans = np.broadcast_to(input_vals[0].reshape(tuple(shape_A)), tuple(new_shape))
        return ans
    def gradient(self, node, output_grad):
        return [auto_sum_op(output_grad, get_shape_op(node.inputs[0])) , zeroslike_op(node.inputs[1])]

class AutoSumOp(Op):
    """using the broadcast system"""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        shape_A = list(input_vals[0].shape)
        shape_B = list(input_vals[1])
        while len(shape_A) > len(shape_B):
            shape_B.insert(0,1)
        axis = []
        for i in range(len(shape_B)):
            if shape_B[i]==1:
                axis.append(i)
        ans = (np.sum(input_vals[0], axis= tuple(axis))).reshape(tuple(input_vals[1]))
        return ans;
    def gradient(self, node, output_grad):
        return [auto_broadcast_op(output_grad, get_shape_op(node.inputs[0])) , zeroslike_op(node.inputs[1])]
class CastOp(Op):
    def __call__(self, node_A, dtype = float64):
        new_node = Op.__call__(self)
        new_node.dtype = dtype
        new_node.inputs = [node_A]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0].astype(node.dtype)

    def gradient(self, node, output_grad):
        return [output_grad]

class ArgmaxOp(Op):
    def __call__(self, node_A, axis=None):
        new_node = Op.__call__(self)
        if axis==None:
            new_node.const_attr = None
        else:
            new_node.const_attr = axis
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        if node.const_attr!=None:
            return np.argmax(input_vals[0], node.const_attr)
        else:
            return np.argmax(input_vals[0])
class EqualOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return np.equal( input_vals[0] , input_vals[1] )

class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        softmax = softmax_func(input_vals[0])
        return -np.sum(input_vals[1] * np.log(softmax), axis=-1, keepdims=True)

    def gradient(self, node, output_grad):
        grad_A = (nn.softmax(node.inputs[0]) -  node.inputs[1]) * output_grad
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        input_vals[0]=np.array(input_vals[0])
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class ReluOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return (np.greater_equal(input_vals[1],0))*input_vals[0]
    def gradient(self, node, output_grad):
        return [relu_op(output_grad, node.inputs[1]) , zeroslike_op(node.inputs[1])]
class Conv2DOp(Op):
    def __call__(self, input, filter, strides, padding = "SAME" , dtype = float32):
        new_node = Op.__call__(self)
        new_node.const_attr = strides
        new_node.dtype = dtype
        new_node.inputs = [input, filter]
        return new_node

    def compute(self, node, input_vals):
        #start = time.time()
        strides = node.const_attr
        ish = list(input_vals[0].shape)
        fsh = list(input_vals[1].shape)
        filter = input_vals[1].astype(float32)
        input = np.zeros((ish[0],ish[1]+fsh[0]-1,ish[2]+fsh[1]-1,ish[3])).astype(float32)
        input[:,fsh[0]//2:fsh[0]//2+ish[1]:1,fsh[1]//2:fsh[1]//2+ish[2]:1,:]+=input_vals[0].astype(float32)
        ish = list(input.shape)
        output = np.zeros([ish[0],(ish[1]-fsh[0])//strides[1]+1,(ish[2]-fsh[1])//strides[2]+1,fsh[3]]).astype(float32)
        osh = output.shape
        for i in range(osh[1]):
            for j in range(osh[2]):
                output[:,i,j,:] = np.matmul(input[:,strides[1]*i:strides[1]*i+fsh[0],strides[2]*j:strides[2]*j+fsh[1],:].reshape((osh[0],-1)),filter.reshape((-1,osh[3])))
        return output
    def gradient(self, node, output_grad):
        return [conv2d_grad_op1(node.inputs[0], node.inputs[1], node.const_attr , output_grad),conv2d_grad_op2(node.inputs[0], node.inputs[1], node.const_attr , output_grad)]

class Conv2DGradientOp1(Op):
    def __call__(self, input, filter, strides, output_grad, padding = "SAME" , dtype = float32):
        new_node = Op.__call__(self)
        new_node.const_attr = strides
        new_node.dtype = dtype
        new_node.inputs = [input, filter, output_grad]
        return new_node

    def compute(self, node, input_vals):
        #start = time.time()
        strides = node.const_attr
        ish = list(input_vals[0].shape)
        fsh = list(input_vals[1].shape)
        filter = input_vals[1].astype(float32)
        ans = np.zeros((ish[0],ish[1]+fsh[0]-1,ish[2]+fsh[1]-1,ish[3])).astype(float32)
        ish = list(ans.shape)
        output_grad = input_vals[2].astype(float32)
        osh = output_grad.shape
        for i in range(osh[1]):
            for j in range(osh[2]):
                #Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
                ans[:,strides[1]*i:strides[1]*i+fsh[0],strides[2]*j:strides[2]*j+fsh[1],:] += np.matmul( output_grad[:,i,j,:], filter.reshape((-1,osh[3])).T).reshape((ish[0],fsh[0],fsh[1],ish[3]))
        
        ish = list(input_vals[0].shape)
        #end = time.time()
        #print("conv2d_grad1")      
        #print(end - start)      
        return ans[:,fsh[0]//2:fsh[0]//2+ish[1]:1,fsh[1]//2:fsh[1]//2+ish[2]:1,:]        
 
class Conv2DGradientOp2(Op):
    def __call__(self, input, filter, strides, output_grad, padding = "SAME" , dtype = float32):
        new_node = Op.__call__(self)
        new_node.const_attr = strides
        new_node.dtype = dtype
        new_node.inputs = [input, filter, output_grad]
        return new_node

    def compute(self, node, input_vals):
        #start = time.time()
        ans = np.zeros(input_vals[1].shape).astype(float32)
        strides = node.const_attr
        ish = list(input_vals[0].shape)
        fsh = list(input_vals[1].shape)
        input = np.zeros((ish[0],ish[1]+fsh[0]-1,ish[2]+fsh[1]-1,ish[3])).astype(float32)
        input[:,fsh[0]//2:fsh[0]//2+ish[1]:1,fsh[1]//2:fsh[1]//2+ish[2]:1,:]+=input_vals[0].astype(float32)
        ish = list(input.shape)
        output_grad = input_vals[2].astype(float32)
        osh = output_grad.shape
        for i in range(osh[1]):
            for j in range(osh[2]):
                #Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
                ans += np.matmul( (input[:,strides[1]*i:strides[1]*i+fsh[0],strides[2]*j:strides[2]*j+fsh[1],:].reshape((osh[0],-1))).T, output_grad[:,i,j,:] ).reshape((fsh[0],fsh[1],fsh[2],fsh[3]))
        #print("conv2d_grad2")  
        #end = time.time()
        #print(end - start)      
        return ans
class MaxPoolOp(Op):
    def __call__(self, value, ksize, strides, padding = "SAME" , dtype = float32):
        new_node = Op.__call__(self)
        new_node.const_attr = (ksize, strides)
        new_node.dtype = dtype
        new_node.inputs = [value]
        return new_node

    def compute(self, node, input_vals):
        strides = node.const_attr[1]
        ksize = node.const_attr[0]
        ish = list(input_vals[0].shape)
        input = input_vals[0]
        output = np.zeros([ish[0],(ish[1]-ksize[1])//strides[1]+1,(ish[2]-ksize[2])//strides[2]+1,ish[3]])
        osh = output.shape
        for i in range(osh[1]):
            for j in range(osh[2]):
                output[:,i,j,:] = np.amax(input[:,i*strides[1]:(i+1)*strides[1],j*strides[1]:(j+1)*strides[1],:],axis=(1,2))
        #end = time.time()  
        #print("max_pool")      
        #print(end - start)      
        return output
        
    def gradient(self, node, output_grad):
        return [max_pool_grad_op(node.inputs[0], output_grad, node.const_attr[0], node.const_attr[1])]
        
class MaxPoolGradOp(Op):
    def __call__(self, value, output_grad, ksize, strides, padding = "SAME" , dtype = float32):
        new_node = Op.__call__(self)
        new_node.const_attr = (ksize, strides)
        new_node.dtype = dtype
        new_node.inputs = [value, output_grad]
        return new_node

    def compute(self, node, input_vals):
        #start = time.time()
        strides = node.const_attr[1]
        ksize = node.const_attr[0]
        ish = list(input_vals[0].shape)
        input = input_vals[0]
        output_grad = input_vals[1]
        output = np.zeros([ish[0],(ish[1]-ksize[1])//strides[1]+1,(ish[2]-ksize[2])//strides[2]+1,ish[3]])
        osh = output_grad.shape
        ans = np.zeros(tuple(ish))
        for i in range(osh[1]):
            for j in range(osh[2]):
                mat1 = input[:,i*strides[1]:(i+1)*strides[1],j*strides[2]:(j+1)*strides[2],:]
                mat3 = np.equal(np.amax(mat1,axis=(1,2),keepdims=True), mat1)
                ans[:,i*strides[1]:(i+1)*strides[1],j*strides[1]:(j+1)*strides[1],:] += mat3 * output_grad[:,i,j,:].reshape(osh[0],1,1,osh[3])
        #end = time.time()  
        #print("max_pool_grad")      
        #print(end - start)      
        return ans
class GetRandOp(Op):
    def __call__(self, shape, keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [shape, keep_prob]
        return new_node

    def compute(self, node, input_vals):
        matA = np.less_equal(np.random.random(tuple(input_vals[0])) , input_vals[1])
        return matA
    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0]), zeroslike_op(node.inputs[1])]

def softmax_func(y):
    expy = np.exp(y - np.max(y, axis=-1, keepdims=True))
    softmax = expy / np.sum(expy, axis=-1, keepdims=True)
    return softmax

class SqrtOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.sqrt(input_vals[0])

    def gradient(self, node, output_grad):
        raise NotImplementedError
        
        
class DropoutOp(Op):
    def __call__(self, x, d , keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [x, d, keep_prob]
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0]*input_vals[1].astype(int32) / input_vals[2]
        
    def gradient(self, node, output_grad):
        return[dropout_op(output_grad, node.inputs[1], node.inputs[2]),zeroslike_op(node.inputs[1]),zeroslike_op(node.inputs[2])]
# Create global singletons of operators.
sqrt_op = SqrtOp()
get_rand_op = GetRandOp()
dropout_op = DropoutOp()
conv2d_op = Conv2DOp()
max_pool_op = MaxPoolOp()
class class_nn:
    conv2d = conv2d_op
    max_pool = max_pool_op
    def softmax(node):
        node_B = exp(node)
        return node_B / broadcast_to( reduce_sum(node_B,reduction_indices=[-1]), get_shape_op(node_B), reduction_indices=[-1])
    def relu(node):
        return relu_op(node,node)
    def softmax_cross_entropy_with_logits(labels = None, logits = None):
        return softmax_cross_entropy_op(logits,labels)
        logits2 = nn.softmax (logits)
        return - reduce_sum(labels*log(logits2), reduction_indices=[-1])
    def dropout(x,keep_pro):
        return dropout_op(x,get_rand_op(get_shape_op(x), keep_pro), keep_pro)
get_shape_op = GetShapeOp()
reshape_op = ReshapeOp()
reshape = ReshapeConstOp()
add_op = AddOp()
sub_op = SubOp()
sub_byconst_op = SubByConstOp()
rsub_byconst_op = RSubByConstOp()
mul_op = MulOp()
exp_op = ExpOp()
log_op = LogOp()
div_op = DivOp()
cast_op = CastOp()
equal_op = EqualOp()
argmax_op = ArgmaxOp()
reduce_sum_op = ReduceSumOp()
reduce_mean_op = ReduceMeanOp()
broadcast_to_op = BroadcastToOp()
broadcast_mean_to_op = BroadcastMeanToOp()
div_byconst_op = DivByConstOp()
rdiv_byconst_op = RDivByConstOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
auto_broadcast_op = AutoBroadcastOp()
auto_sum_op  =  AutoSumOp()
conv2d_grad_op1 = Conv2DGradientOp1()
conv2d_grad_op2 = Conv2DGradientOp2()
max_pool_grad_op = MaxPoolGradOp()
nn = class_nn
relu_op = ReluOp()
softmax_cross_entropy_op = SoftmaxCrossEntropyOp()
class Executor:
    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(self.eval_node_list)
        for node in topo_order :
            if isinstance(node.op, PlaceholderOp):
                continue 
            if not(node in node_to_val_map):
                input_vals1=[]
                for inp in node.inputs:
                    input_vals1.append( node_to_val_map[inp] )
                node_to_val_map[node] = node.op.compute(node, input_vals1)
        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results

def gradients(output_node, node_list):
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    reverse_topo_order = list(reversed(find_topo_sort([output_node])))
    for node in reverse_topo_order:
        if not(node in node_to_output_grad):
            sum_node =  sum_node_list (node_to_output_grads_list[node]) 
            grad = node.op.gradient(node, sum_node)
            node_to_output_grad[node] = sum_node
            for i in range(len(node.inputs)):
                if (not(node.inputs[i] in node_to_output_grads_list)):
                    node_to_output_grads_list[node.inputs[i]]=[]
                node_to_output_grads_list[node.inputs[i]].append(grad[i])
    del reverse_topo_order
    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

class Session:
    def __enter__(self):
        return self
    def __exit__(self, e_t, e_v, t_b):
        assert True
    def __init__(self):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = []
    
    def run(self, eval_node , feed_dict = {}):
        if isinstance(eval_node, global_variables_initializer_type):
            return
        bool_is_list = True
        if not isinstance(eval_node, list):
            bool_is_list = False
            eval_node = [eval_node]
        eval_list = eval_node
        eval_node = simple_list(eval_node)
        return_dict = {}
        input_node = []
        for node in eval_node:
            if isinstance(node, Node):
                input_node.append(node)
            elif isinstance(node, assign_type):
                input_node.append(node.val)
            elif True:
                input_node.append(0)
                print(node.data)
                assert False
        result_list = self.run1(input_node, feed_dict)
        return_list = []
        for i in range(len(eval_node)):
            if isinstance(eval_node[i],Node):
                return_dict[eval_node[i]] = result_list[i]
            elif isinstance(eval_node[i], assign_type):
                global_variables[eval_node[i].node] = result_list[i]
        for node in eval_list:
            if isinstance(node,Node):
                return_list.append(return_dict[node])
            else :
                return_list.append(None)
        if len(return_list)==0:
            return
        if not bool_is_list:
            return return_list[0]
        return return_list
    def run1(self, eval_node , feed_dict = {}):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.
        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        self.eval_node_list = eval_node 
        for i in feed_dict:
            feed_dict[i] = np.array(feed_dict[i]).astype(i.dtype)
        #print(self.eval_node_list)
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_start_node = []
        for node in self.eval_node_list:
            if isinstance(node,Node):
                topo_start_node.append(node)
        if topo_start_node==[]:
            return eval_node
        topo_order = find_topo_sort(topo_start_node)
        for node in topo_order :
            if isinstance(node.op, PlaceholderOp):
                if not(node in node_to_val_map) and (node in global_variables):
                    node_to_val_map[node] = global_variables[node]
                else:
                    # to make the data lighter and faster
                    # take data randomly
                    sh = node_to_val_map[node].shape
                    if (len(sh)>0 and sh[0]>1000):
                        #print(sh[0])
                        node_to_val_map[node] = node_to_val_map[node][0:1000]
                    elif (len(sh)>0 and sh[0]>50):
                        #print(sh[0])
                        node_to_val_map[node] = node_to_val_map[node][0:50]
                continue 
            if not(node in node_to_val_map):
                input_vals1=[]
                for inp in node.inputs:
                    input_vals1.append( node_to_val_map[inp] )
                node_to_val_map[node] = node.op.compute(node, input_vals1)
        # Collect node values.
        node_val_results = []
        for node in self.eval_node_list:
            if isinstance(node, Node):
                node_val_results.append(node_to_val_map[node])
            else:
                node_val_results.append(node)
        return node_val_results
##############################
####### Helper Methods #######
##############################
def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited = set()
    topo_order = []
    #print(node_list)
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)