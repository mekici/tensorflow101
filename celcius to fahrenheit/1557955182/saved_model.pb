�_
�
�

:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�"serve*1.13.12
b'unknown'8�N
q
dense_14_inputPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
0dense_14/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_14/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
.dense_14/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_14/kernel*
valueB
 *׳ݿ*
dtype0*
_output_shapes
: 
�
.dense_14/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_14/kernel*
valueB
 *׳�?*
dtype0*
_output_shapes
: 
�
8dense_14/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_14/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_14/kernel*
dtype0*
_output_shapes

:
�
.dense_14/kernel/Initializer/random_uniform/subSub.dense_14/kernel/Initializer/random_uniform/max.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes
: 
�
.dense_14/kernel/Initializer/random_uniform/mulMul8dense_14/kernel/Initializer/random_uniform/RandomUniform.dense_14/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes

:
�
*dense_14/kernel/Initializer/random_uniformAdd.dense_14/kernel/Initializer/random_uniform/mul.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes

:
�
dense_14/kernelVarHandleOp*
shape
:*
dtype0*
_output_shapes
: * 
shared_namedense_14/kernel*"
_class
loc:@dense_14/kernel
o
0dense_14/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/kernel*
_output_shapes
: 
�
dense_14/kernel/AssignAssignVariableOpdense_14/kernel*dense_14/kernel/Initializer/random_uniform*"
_class
loc:@dense_14/kernel*
dtype0
�
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*"
_class
loc:@dense_14/kernel*
dtype0*
_output_shapes

:
�
dense_14/bias/Initializer/zerosConst* 
_class
loc:@dense_14/bias*
valueB*    *
dtype0*
_output_shapes
:
�
dense_14/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_14/bias* 
_class
loc:@dense_14/bias*
shape:
k
.dense_14/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/bias*
_output_shapes
: 
�
dense_14/bias/AssignAssignVariableOpdense_14/biasdense_14/bias/Initializer/zeros* 
_class
loc:@dense_14/bias*
dtype0
�
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias* 
_class
loc:@dense_14/bias*
dtype0*
_output_shapes
:
n
dense_14/MatMul/ReadVariableOpReadVariableOpdense_14/kernel*
dtype0*
_output_shapes

:
{
dense_14/MatMulMatMuldense_14_inputdense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
i
dense_14/BiasAdd/ReadVariableOpReadVariableOpdense_14/bias*
dtype0*
_output_shapes
:

dense_14/BiasAddBiasAdddense_14/MatMuldense_14/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:���������
-
predict/group_depsNoOp^dense_14/BiasAdd
U
ConstConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
\
Const_3Const"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
W
Const_4Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_5Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
�
RestoreV2/tensor_namesConst*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
c
RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
~
	RestoreV2	RestoreV2Const_3RestoreV2/tensor_namesRestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
L
AssignVariableOpAssignVariableOpdense_14/kernelIdentity*
dtype0
�
RestoreV2_1/tensor_namesConst*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
RestoreV2_1	RestoreV2Const_3RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
N
AssignVariableOp_1AssignVariableOpdense_14/bias
Identity_1*
dtype0
Q
VarIsInitializedOpVarIsInitializedOpdense_14/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_1VarIsInitializedOpdense_14/bias*
_output_shapes
: 
<
initNoOp^dense_14/bias/Assign^dense_14/kernel/Assign
W
Const_7Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
�
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:
y
SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B B *
dtype0*
_output_shapes
:
�
SaveV2SaveV2Const_8SaveV2/tensor_namesSaveV2/shape_and_slicesConst_4Const_5Const_6#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpConst_7"/device:CPU:0*
dtypes

2
X

Identity_2IdentityConst_8^SaveV2"/device:CPU:0*
T0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
�
save/SaveV2/tensor_namesConst*�
value�B�B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
m
save/SaveV2/shape_and_slicesConst*
valueBB B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2/tensors_0Const*�
value�B� B�{"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"activation": "linear", "activity_regularizer": null, "batch_input_shape": [null, 1], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_14", "trainable": true, "units": 1, "use_bias": true}}], "name": "sequential_14"}}*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensors_1Const*�
value�B� B�{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1], "dtype": "float32", "name": "dense_14_input", "sparse": false}}*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensors_2Const*
dtype0*
_output_shapes
: *�
value�B� B�{"class_name": "Dense", "config": {"activation": "linear", "activity_regularizer": null, "batch_input_shape": [null, 1], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_14", "trainable": true, "units": 1, "use_bias": true}}
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/SaveV2/tensors_0save/SaveV2/tensors_1save/SaveV2/tensors_2!dense_14/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp*
dtypes	
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes	
2*(
_output_shapes
:::::

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp
N
save/IdentityIdentitysave/RestoreV2:3*
_output_shapes
:*
T0
T
save/AssignVariableOpAssignVariableOpdense_14/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:4*
_output_shapes
:*
T0
Z
save/AssignVariableOp_1AssignVariableOpdense_14/kernelsave/Identity_1*
dtype0
r
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1
^save/NoOp^save/NoOp_1^save/NoOp_2

init_1NoOp"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"�
trainable_variables��
�
dense_14/kernel:0dense_14/kernel/Assign%dense_14/kernel/Read/ReadVariableOp:0(2,dense_14/kernel/Initializer/random_uniform:08
s
dense_14/bias:0dense_14/bias/Assign#dense_14/bias/Read/ReadVariableOp:0(2!dense_14/bias/Initializer/zeros:08"�
	variables��
�
dense_14/kernel:0dense_14/kernel/Assign%dense_14/kernel/Read/ReadVariableOp:0(2,dense_14/kernel/Initializer/random_uniform:08
s
dense_14/bias:0dense_14/bias/Assign#dense_14/bias/Read/ReadVariableOp:0(2!dense_14/bias/Initializer/zeros:08*@
__saved_model_init_op'%
__saved_model_init_op
init_1*�
serving_default�
9
dense_14_input'
dense_14_input:0���������5
dense_14)
dense_14/BiasAdd:0���������tensorflow/serving/predict