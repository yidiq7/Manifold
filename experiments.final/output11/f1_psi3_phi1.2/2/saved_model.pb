čÉ
Ķ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18
m
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	į*
shared_name
Variable
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	į*
dtype0

NoOpNoOp
¾
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ł
valueļBģ Bå

biholomorphic_k2

layer1
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
		variables

trainable_variables
	keras_api
Y
w
regularization_losses
	variables
trainable_variables
	keras_api
 

0

0
­
regularization_losses

layers
non_trainable_variables
metrics
layer_metrics
layer_regularization_losses
	variables
trainable_variables
 
 
 
 
­
regularization_losses

layers
non_trainable_variables
metrics
layer_metrics
layer_regularization_losses
		variables

trainable_variables
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses

layers
non_trainable_variables
metrics
layer_metrics
layer_regularization_losses
	variables
trainable_variables

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
Į
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_839172
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
½
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_839316
¤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_839329ßü
«

K__inference_width_one_dense_layer_call_and_return_conditional_losses_839283

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	į*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :’’’’’’’’’’’’’’’’’’::X T
0
_output_shapes
:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
é

z
"__inference__traced_restore_839329
file_prefix
assignvariableop_variable

identity_2¢AssignVariableOpÉ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesµ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp9
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp{

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_1m

Identity_2IdentityIdentity_1:output:0^AssignVariableOp*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*
_input_shapes
: :2$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ß
v
0__inference_width_one_dense_layer_call_fn_839290

inputs
unknown
identity¢StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_width_one_dense_layer_call_and_return_conditional_losses_8391412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :’’’’’’’’’’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ź
µ
O__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_839155
input_1
width_one_dense_839150
identity¢'width_one_dense/StatefulPartitionedCallł
 biholomorphic_k2/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_8391262"
 biholomorphic_k2/PartitionedCallĆ
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall)biholomorphic_k2/PartitionedCall:output:0width_one_dense_839150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_width_one_dense_layer_call_and_return_conditional_losses_8391412)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Log
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:2R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
÷p
h
L__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_839271

inputs
identity
einsum/EinsumEinsuminputsinputs*
N*
T0*+
_output_shapes
:’’’’’’’’’*
equation
ai,aj->aij2
einsum/Einsumv
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart/num_lower
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’2
MatrixBandPart/num_upperĘ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
	transposeP
Abs
ComplexAbstranspose:y:0*'
_output_shapes
:’’’’’’’’’2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:2
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:2
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:2

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:2	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices¢
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2ø
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1i
boolean_mask/Shape_2Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2ø
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat/values_1v
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/concat/axisü
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:’’’’’’’’’2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisū
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Reshape_1/shape~
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:’’’’’’’’’2
ConjØ
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:’’’’’’’’’*
equation
ai,aj->aij2
einsum_1/Einsumz
MatrixBandPart_1/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_1/num_lower
MatrixBandPart_1/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’2
MatrixBandPart_1/num_upperŠ
MatrixBandPart_1MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
MatrixBandPart_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’į   2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_1:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’į2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:’’’’’’’’’į2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:’’’’’’’’’į2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’Ā2
concatu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transposeconcat:output:0transpose_2/perm:output:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2
transpose_2Y
Abs_1Abstranspose_2:y:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2
Abs_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesh
Sum_1Sum	Abs_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:Ā2
Sum_1Y
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2

Less_1/ya
Less_1LessSum_1:output:0Less_1/y:output:0*
T0*
_output_shapes	
:Ā2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:Ā2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:Ā2
	Squeeze_1k
boolean_mask_1/ShapeShapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask_1/strided_slice/stack
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_1
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_2Ø
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask_1/strided_slice
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2'
%boolean_mask_1/Prod/reduction_indicesŖ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodo
boolean_mask_1/Shape_1Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_1
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask_1/strided_slice_1/stack
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_1/stack_1
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_1/stack_2Ä
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1o
boolean_mask_1/Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_2
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice_2/stack
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_2/stack_1
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_2/stack_2Ä
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
boolean_mask_1/strided_slice_2
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2 
boolean_mask_1/concat/values_1z
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/concat/axis
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask_1/concat
boolean_mask_1/ReshapeReshapetranspose_2:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2 
boolean_mask_1/Reshape_1/shape¢
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:Ā2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:’’’’’’’’’2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
2
boolean_mask_1/Squeeze~
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/GatherV2/axis
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
boolean_mask_1/GatherV2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose boolean_mask_1/GatherV2:output:0transpose_3/perm:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
transpose_3l
IdentityIdentitytranspose_3:y:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ų
{
4__inference_outer_product_nn_k2_layer_call_fn_839163
input_1
unknown
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_8391552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ūĖ

!__inference__wrapped_model_839023
input_1F
Bouter_product_nn_k2_width_one_dense_matmul_readvariableop_resource
identityŻ
2outer_product_nn_k2/biholomorphic_k2/einsum/EinsumEinsuminput_1input_1*
N*
T0*+
_output_shapes
:’’’’’’’’’*
equation
ai,aj->aij24
2outer_product_nn_k2/biholomorphic_k2/einsum/EinsumĄ
=outer_product_nn_k2/biholomorphic_k2/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2?
=outer_product_nn_k2/biholomorphic_k2/MatrixBandPart/num_lowerÉ
=outer_product_nn_k2/biholomorphic_k2/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’2?
=outer_product_nn_k2/biholomorphic_k2/MatrixBandPart/num_upper’
3outer_product_nn_k2/biholomorphic_k2/MatrixBandPartMatrixBandPart;outer_product_nn_k2/biholomorphic_k2/einsum/Einsum:output:0Fouter_product_nn_k2/biholomorphic_k2/MatrixBandPart/num_lower:output:0Fouter_product_nn_k2/biholomorphic_k2/MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:’’’’’’’’’25
3outer_product_nn_k2/biholomorphic_k2/MatrixBandPart¹
2outer_product_nn_k2/biholomorphic_k2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   24
2outer_product_nn_k2/biholomorphic_k2/Reshape/shape
,outer_product_nn_k2/biholomorphic_k2/ReshapeReshape:outer_product_nn_k2/biholomorphic_k2/MatrixBandPart:band:0;outer_product_nn_k2/biholomorphic_k2/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2.
,outer_product_nn_k2/biholomorphic_k2/Reshape»
3outer_product_nn_k2/biholomorphic_k2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       25
3outer_product_nn_k2/biholomorphic_k2/transpose/perm
.outer_product_nn_k2/biholomorphic_k2/transpose	Transpose5outer_product_nn_k2/biholomorphic_k2/Reshape:output:0<outer_product_nn_k2/biholomorphic_k2/transpose/perm:output:0*
T0*'
_output_shapes
:’’’’’’’’’20
.outer_product_nn_k2/biholomorphic_k2/transposeæ
(outer_product_nn_k2/biholomorphic_k2/Abs
ComplexAbs2outer_product_nn_k2/biholomorphic_k2/transpose:y:0*'
_output_shapes
:’’’’’’’’’2*
(outer_product_nn_k2/biholomorphic_k2/Absŗ
:outer_product_nn_k2/biholomorphic_k2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:outer_product_nn_k2/biholomorphic_k2/Sum/reduction_indicesó
(outer_product_nn_k2/biholomorphic_k2/SumSum,outer_product_nn_k2/biholomorphic_k2/Abs:y:0Couter_product_nn_k2/biholomorphic_k2/Sum/reduction_indices:output:0*
T0*
_output_shapes
:2*
(outer_product_nn_k2/biholomorphic_k2/Sum
+outer_product_nn_k2/biholomorphic_k2/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+outer_product_nn_k2/biholomorphic_k2/Less/yģ
)outer_product_nn_k2/biholomorphic_k2/LessLess1outer_product_nn_k2/biholomorphic_k2/Sum:output:04outer_product_nn_k2/biholomorphic_k2/Less/y:output:0*
T0*
_output_shapes
:2+
)outer_product_nn_k2/biholomorphic_k2/Less»
/outer_product_nn_k2/biholomorphic_k2/LogicalNot
LogicalNot-outer_product_nn_k2/biholomorphic_k2/Less:z:0*
_output_shapes
:21
/outer_product_nn_k2/biholomorphic_k2/LogicalNotĮ
,outer_product_nn_k2/biholomorphic_k2/SqueezeSqueeze3outer_product_nn_k2/biholomorphic_k2/LogicalNot:y:0*
T0
*
_output_shapes
:2.
,outer_product_nn_k2/biholomorphic_k2/SqueezeŌ
7outer_product_nn_k2/biholomorphic_k2/boolean_mask/ShapeShape2outer_product_nn_k2/biholomorphic_k2/transpose:y:0*
T0*
_output_shapes
:29
7outer_product_nn_k2/biholomorphic_k2/boolean_mask/ShapeŲ
Eouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Eouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stackÜ
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stack_1Ü
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stack_2ś
?outer_product_nn_k2/biholomorphic_k2/boolean_mask/strided_sliceStridedSlice@outer_product_nn_k2/biholomorphic_k2/boolean_mask/Shape:output:0Nouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stack:output:0Pouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stack_1:output:0Pouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2A
?outer_product_nn_k2/biholomorphic_k2/boolean_mask/strided_sliceŽ
Houter_product_nn_k2/biholomorphic_k2/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2J
Houter_product_nn_k2/biholomorphic_k2/boolean_mask/Prod/reduction_indices¶
6outer_product_nn_k2/biholomorphic_k2/boolean_mask/ProdProdHouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice:output:0Qouter_product_nn_k2/biholomorphic_k2/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 28
6outer_product_nn_k2/biholomorphic_k2/boolean_mask/ProdŲ
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/Shape_1Shape2outer_product_nn_k2/biholomorphic_k2/transpose:y:0*
T0*
_output_shapes
:2;
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/Shape_1Ü
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stacką
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stack_1ą
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stack_2
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1StridedSliceBouter_product_nn_k2/biholomorphic_k2/boolean_mask/Shape_1:output:0Pouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stack:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stack_1:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2C
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1Ų
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/Shape_2Shape2outer_product_nn_k2/biholomorphic_k2/transpose:y:0*
T0*
_output_shapes
:2;
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/Shape_2Ü
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2I
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stacką
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stack_1ą
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stack_2
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2StridedSliceBouter_product_nn_k2/biholomorphic_k2/boolean_mask/Shape_2:output:0Pouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stack:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stack_1:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2C
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2ż
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/concat/values_1Pack?outer_product_nn_k2/biholomorphic_k2/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2C
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/concat/values_1Ą
=outer_product_nn_k2/biholomorphic_k2/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=outer_product_nn_k2/biholomorphic_k2/boolean_mask/concat/axisŚ
8outer_product_nn_k2/biholomorphic_k2/boolean_mask/concatConcatV2Jouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_1:output:0Jouter_product_nn_k2/biholomorphic_k2/boolean_mask/concat/values_1:output:0Jouter_product_nn_k2/biholomorphic_k2/boolean_mask/strided_slice_2:output:0Fouter_product_nn_k2/biholomorphic_k2/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2:
8outer_product_nn_k2/biholomorphic_k2/boolean_mask/concatŖ
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/ReshapeReshape2outer_product_nn_k2/biholomorphic_k2/transpose:y:0Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/concat:output:0*
T0*'
_output_shapes
:’’’’’’’’’2;
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/ReshapeŁ
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2C
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask/Reshape_1/shape­
;outer_product_nn_k2/biholomorphic_k2/boolean_mask/Reshape_1Reshape5outer_product_nn_k2/biholomorphic_k2/Squeeze:output:0Jouter_product_nn_k2/biholomorphic_k2/boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:2=
;outer_product_nn_k2/biholomorphic_k2/boolean_mask/Reshape_1ź
7outer_product_nn_k2/biholomorphic_k2/boolean_mask/WhereWhereDouter_product_nn_k2/biholomorphic_k2/boolean_mask/Reshape_1:output:0*'
_output_shapes
:’’’’’’’’’29
7outer_product_nn_k2/biholomorphic_k2/boolean_mask/Where
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/SqueezeSqueeze?outer_product_nn_k2/biholomorphic_k2/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
2;
9outer_product_nn_k2/biholomorphic_k2/boolean_mask/SqueezeÄ
?outer_product_nn_k2/biholomorphic_k2/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?outer_product_nn_k2/biholomorphic_k2/boolean_mask/GatherV2/axis“
:outer_product_nn_k2/biholomorphic_k2/boolean_mask/GatherV2GatherV2Bouter_product_nn_k2/biholomorphic_k2/boolean_mask/Reshape:output:0Bouter_product_nn_k2/biholomorphic_k2/boolean_mask/Squeeze:output:0Houter_product_nn_k2/biholomorphic_k2/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2<
:outer_product_nn_k2/biholomorphic_k2/boolean_mask/GatherV2æ
5outer_product_nn_k2/biholomorphic_k2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5outer_product_nn_k2/biholomorphic_k2/transpose_1/perm±
0outer_product_nn_k2/biholomorphic_k2/transpose_1	TransposeCouter_product_nn_k2/biholomorphic_k2/boolean_mask/GatherV2:output:0>outer_product_nn_k2/biholomorphic_k2/transpose_1/perm:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’22
0outer_product_nn_k2/biholomorphic_k2/transpose_1½
4outer_product_nn_k2/biholomorphic_k2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   26
4outer_product_nn_k2/biholomorphic_k2/Reshape_1/shape
.outer_product_nn_k2/biholomorphic_k2/Reshape_1Reshape4outer_product_nn_k2/biholomorphic_k2/transpose_1:y:0=outer_product_nn_k2/biholomorphic_k2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’20
.outer_product_nn_k2/biholomorphic_k2/Reshape_1Ą
)outer_product_nn_k2/biholomorphic_k2/ConjConj7outer_product_nn_k2/biholomorphic_k2/Reshape_1:output:0*'
_output_shapes
:’’’’’’’’’2+
)outer_product_nn_k2/biholomorphic_k2/Conj¼
4outer_product_nn_k2/biholomorphic_k2/einsum_1/EinsumEinsum7outer_product_nn_k2/biholomorphic_k2/Reshape_1:output:02outer_product_nn_k2/biholomorphic_k2/Conj:output:0*
N*
T0*+
_output_shapes
:’’’’’’’’’*
equation
ai,aj->aij26
4outer_product_nn_k2/biholomorphic_k2/einsum_1/EinsumÄ
?outer_product_nn_k2/biholomorphic_k2/MatrixBandPart_1/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2A
?outer_product_nn_k2/biholomorphic_k2/MatrixBandPart_1/num_lowerĶ
?outer_product_nn_k2/biholomorphic_k2/MatrixBandPart_1/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’2A
?outer_product_nn_k2/biholomorphic_k2/MatrixBandPart_1/num_upper
5outer_product_nn_k2/biholomorphic_k2/MatrixBandPart_1MatrixBandPart=outer_product_nn_k2/biholomorphic_k2/einsum_1/Einsum:output:0Houter_product_nn_k2/biholomorphic_k2/MatrixBandPart_1/num_lower:output:0Houter_product_nn_k2/biholomorphic_k2/MatrixBandPart_1/num_upper:output:0*
T0*+
_output_shapes
:’’’’’’’’’27
5outer_product_nn_k2/biholomorphic_k2/MatrixBandPart_1½
4outer_product_nn_k2/biholomorphic_k2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’į   26
4outer_product_nn_k2/biholomorphic_k2/Reshape_2/shape
.outer_product_nn_k2/biholomorphic_k2/Reshape_2Reshape<outer_product_nn_k2/biholomorphic_k2/MatrixBandPart_1:band:0=outer_product_nn_k2/biholomorphic_k2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’į20
.outer_product_nn_k2/biholomorphic_k2/Reshape_2Į
)outer_product_nn_k2/biholomorphic_k2/RealReal7outer_product_nn_k2/biholomorphic_k2/Reshape_2:output:0*(
_output_shapes
:’’’’’’’’’į2+
)outer_product_nn_k2/biholomorphic_k2/RealĮ
)outer_product_nn_k2/biholomorphic_k2/ImagImag7outer_product_nn_k2/biholomorphic_k2/Reshape_2:output:0*(
_output_shapes
:’’’’’’’’’į2+
)outer_product_nn_k2/biholomorphic_k2/Imag¦
0outer_product_nn_k2/biholomorphic_k2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :22
0outer_product_nn_k2/biholomorphic_k2/concat/axisÅ
+outer_product_nn_k2/biholomorphic_k2/concatConcatV22outer_product_nn_k2/biholomorphic_k2/Real:output:02outer_product_nn_k2/biholomorphic_k2/Imag:output:09outer_product_nn_k2/biholomorphic_k2/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’Ā2-
+outer_product_nn_k2/biholomorphic_k2/concatæ
5outer_product_nn_k2/biholomorphic_k2/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5outer_product_nn_k2/biholomorphic_k2/transpose_2/perm
0outer_product_nn_k2/biholomorphic_k2/transpose_2	Transpose4outer_product_nn_k2/biholomorphic_k2/concat:output:0>outer_product_nn_k2/biholomorphic_k2/transpose_2/perm:output:0*
T0*(
_output_shapes
:Ā’’’’’’’’’22
0outer_product_nn_k2/biholomorphic_k2/transpose_2Č
*outer_product_nn_k2/biholomorphic_k2/Abs_1Abs4outer_product_nn_k2/biholomorphic_k2/transpose_2:y:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2,
*outer_product_nn_k2/biholomorphic_k2/Abs_1¾
<outer_product_nn_k2/biholomorphic_k2/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<outer_product_nn_k2/biholomorphic_k2/Sum_1/reduction_indicesü
*outer_product_nn_k2/biholomorphic_k2/Sum_1Sum.outer_product_nn_k2/biholomorphic_k2/Abs_1:y:0Eouter_product_nn_k2/biholomorphic_k2/Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:Ā2,
*outer_product_nn_k2/biholomorphic_k2/Sum_1£
-outer_product_nn_k2/biholomorphic_k2/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-outer_product_nn_k2/biholomorphic_k2/Less_1/yõ
+outer_product_nn_k2/biholomorphic_k2/Less_1Less3outer_product_nn_k2/biholomorphic_k2/Sum_1:output:06outer_product_nn_k2/biholomorphic_k2/Less_1/y:output:0*
T0*
_output_shapes	
:Ā2-
+outer_product_nn_k2/biholomorphic_k2/Less_1Ā
1outer_product_nn_k2/biholomorphic_k2/LogicalNot_1
LogicalNot/outer_product_nn_k2/biholomorphic_k2/Less_1:z:0*
_output_shapes	
:Ā23
1outer_product_nn_k2/biholomorphic_k2/LogicalNot_1Č
.outer_product_nn_k2/biholomorphic_k2/Squeeze_1Squeeze5outer_product_nn_k2/biholomorphic_k2/LogicalNot_1:y:0*
T0
*
_output_shapes	
:Ā20
.outer_product_nn_k2/biholomorphic_k2/Squeeze_1Ś
9outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/ShapeShape4outer_product_nn_k2/biholomorphic_k2/transpose_2:y:0*
T0*
_output_shapes
:2;
9outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/ShapeÜ
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stacką
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stack_1ą
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stack_2
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_sliceStridedSliceBouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Shape:output:0Pouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stack:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stack_1:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2C
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_sliceā
Jouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Prod/reduction_indices¾
8outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/ProdProdJouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice:output:0Souter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2:
8outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/ProdŽ
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/Shape_1Shape4outer_product_nn_k2/biholomorphic_k2/transpose_2:y:0*
T0*
_output_shapes
:2=
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/Shape_1ą
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stackä
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2M
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stack_1ä
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stack_2¢
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1StridedSliceDouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Shape_1:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stack:output:0Touter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stack_1:output:0Touter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2E
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1Ž
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/Shape_2Shape4outer_product_nn_k2/biholomorphic_k2/transpose_2:y:0*
T0*
_output_shapes
:2=
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/Shape_2ą
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2K
Iouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stackä
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2M
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stack_1ä
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Kouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stack_2¢
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2StridedSliceDouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Shape_2:output:0Router_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stack:output:0Touter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stack_1:output:0Touter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2E
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat/values_1PackAouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2E
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat/values_1Ä
?outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat/axisę
:outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/concatConcatV2Louter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_1:output:0Louter_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat/values_1:output:0Louter_product_nn_k2/biholomorphic_k2/boolean_mask_1/strided_slice_2:output:0Houter_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat³
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/ReshapeReshape4outer_product_nn_k2/biholomorphic_k2/transpose_2:y:0Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2=
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/ReshapeŻ
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2E
Couter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Reshape_1/shape¶
=outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/Reshape_1Reshape7outer_product_nn_k2/biholomorphic_k2/Squeeze_1:output:0Louter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:Ā2?
=outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/Reshape_1š
9outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/WhereWhereFouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:’’’’’’’’’2;
9outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/Where
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/SqueezeSqueezeAouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
2=
;outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/SqueezeČ
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Aouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/GatherV2/axis¾
<outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/GatherV2GatherV2Douter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Reshape:output:0Douter_product_nn_k2/biholomorphic_k2/boolean_mask_1/Squeeze:output:0Jouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2>
<outer_product_nn_k2/biholomorphic_k2/boolean_mask_1/GatherV2æ
5outer_product_nn_k2/biholomorphic_k2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5outer_product_nn_k2/biholomorphic_k2/transpose_3/perm³
0outer_product_nn_k2/biholomorphic_k2/transpose_3	TransposeEouter_product_nn_k2/biholomorphic_k2/boolean_mask_1/GatherV2:output:0>outer_product_nn_k2/biholomorphic_k2/transpose_3/perm:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’22
0outer_product_nn_k2/biholomorphic_k2/transpose_3ś
9outer_product_nn_k2/width_one_dense/MatMul/ReadVariableOpReadVariableOpBouter_product_nn_k2_width_one_dense_matmul_readvariableop_resource*
_output_shapes
:	į*
dtype02;
9outer_product_nn_k2/width_one_dense/MatMul/ReadVariableOp
*outer_product_nn_k2/width_one_dense/MatMulMatMul4outer_product_nn_k2/biholomorphic_k2/transpose_3:y:0Aouter_product_nn_k2/width_one_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2,
*outer_product_nn_k2/width_one_dense/MatMul”
outer_product_nn_k2/LogLog4outer_product_nn_k2/width_one_dense/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
outer_product_nn_k2/Logo
IdentityIdentityouter_product_nn_k2/Log:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
«

K__inference_width_one_dense_layer_call_and_return_conditional_losses_839141

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	į*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :’’’’’’’’’’’’’’’’’’::X T
0
_output_shapes
:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
 

__inference__traced_save_839316
file_prefix'
#savev2_variable_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e598ff6ccb5a4c958d7a0a4f6c1e19f3/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĆ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slicesą
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*"
_input_shapes
: :	į: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	į:

_output_shapes
: 
·
M
1__inference_biholomorphic_k2_layer_call_fn_839276

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_8391262
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

k
$__inference_signature_wrapper_839172
input_1
unknown
identity¢StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_8390232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
÷p
h
L__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_839126

inputs
identity
einsum/EinsumEinsuminputsinputs*
N*
T0*+
_output_shapes
:’’’’’’’’’*
equation
ai,aj->aij2
einsum/Einsumv
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart/num_lower
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’2
MatrixBandPart/num_upperĘ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
	transposeP
Abs
ComplexAbstranspose:y:0*'
_output_shapes
:’’’’’’’’’2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:2
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:2
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:2

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:2	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices¢
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2ø
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1i
boolean_mask/Shape_2Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2ø
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat/values_1v
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/concat/axisü
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:’’’’’’’’’2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisū
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Reshape_1/shape~
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:’’’’’’’’’2
ConjØ
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:’’’’’’’’’*
equation
ai,aj->aij2
einsum_1/Einsumz
MatrixBandPart_1/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_1/num_lower
MatrixBandPart_1/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’2
MatrixBandPart_1/num_upperŠ
MatrixBandPart_1MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
MatrixBandPart_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’į   2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_1:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’į2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:’’’’’’’’’į2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:’’’’’’’’’į2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’Ā2
concatu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transposeconcat:output:0transpose_2/perm:output:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2
transpose_2Y
Abs_1Abstranspose_2:y:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2
Abs_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesh
Sum_1Sum	Abs_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:Ā2
Sum_1Y
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2

Less_1/ya
Less_1LessSum_1:output:0Less_1/y:output:0*
T0*
_output_shapes	
:Ā2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:Ā2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:Ā2
	Squeeze_1k
boolean_mask_1/ShapeShapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask_1/strided_slice/stack
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_1
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_2Ø
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask_1/strided_slice
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2'
%boolean_mask_1/Prod/reduction_indicesŖ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodo
boolean_mask_1/Shape_1Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_1
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask_1/strided_slice_1/stack
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_1/stack_1
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_1/stack_2Ä
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1o
boolean_mask_1/Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_2
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice_2/stack
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_2/stack_1
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_2/stack_2Ä
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
boolean_mask_1/strided_slice_2
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2 
boolean_mask_1/concat/values_1z
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/concat/axis
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask_1/concat
boolean_mask_1/ReshapeReshapetranspose_2:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:Ā’’’’’’’’’2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2 
boolean_mask_1/Reshape_1/shape¢
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:Ā2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:’’’’’’’’’2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
2
boolean_mask_1/Squeeze~
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/GatherV2/axis
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
boolean_mask_1/GatherV2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose boolean_mask_1/GatherV2:output:0transpose_3/perm:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
transpose_3l
IdentityIdentitytranspose_3:y:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0’’’’’’’’’<
output_10
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:¬2
õ
biholomorphic_k2

layer1
regularization_losses
	variables
trainable_variables
	keras_api

signatures
* &call_and_return_all_conditional_losses
!__call__
"_default_save_signature"
_tf_keras_modelż{"class_name": "OuterProductNN_k2", "name": "outer_product_nn_k2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "OuterProductNN_k2"}}
Ķ
regularization_losses
		variables

trainable_variables
	keras_api
*#&call_and_return_all_conditional_losses
$__call__"¾
_tf_keras_layer¤{"class_name": "Biholomorphic_k2", "name": "biholomorphic_k2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic_k2", "trainable": true, "dtype": "float32"}}
µ
w
regularization_losses
	variables
trainable_variables
	keras_api
*%&call_and_return_all_conditional_losses
&__call__"
_tf_keras_layer{"class_name": "WidthOneDense", "name": "width_one_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
Ź
regularization_losses

layers
non_trainable_variables
metrics
layer_metrics
layer_regularization_losses
	variables
trainable_variables
!__call__
"_default_save_signature
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
,
'serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses

layers
non_trainable_variables
metrics
layer_metrics
layer_regularization_losses
		variables

trainable_variables
$__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
:	į2Variable
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
regularization_losses

layers
non_trainable_variables
metrics
layer_metrics
layer_regularization_losses
	variables
trainable_variables
&__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
2
O__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_839155Ę
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
2’
4__inference_outer_product_nn_k2_layer_call_fn_839163Ę
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
ß2Ü
!__inference__wrapped_model_839023¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
ö2ó
L__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_839271¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ū2Ų
1__inference_biholomorphic_k2_layer_call_fn_839276¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
õ2ņ
K__inference_width_one_dense_layer_call_and_return_conditional_losses_839283¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ś2×
0__inference_width_one_dense_layer_call_fn_839290¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
3B1
$__inference_signature_wrapper_839172input_1
!__inference__wrapped_model_839023j0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’±
L__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_839271a/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’’’’’’’’’’
 
1__inference_biholomorphic_k2_layer_call_fn_839276T/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’’’’’’’’’’Æ
O__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_839155\0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 
4__inference_outer_product_nn_k2_layer_call_fn_839163O0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "’’’’’’’’’
$__inference_signature_wrapper_839172u;¢8
¢ 
1Ŗ.
,
input_1!
input_1’’’’’’’’’"3Ŗ0
.
output_1"
output_1’’’’’’’’’³
K__inference_width_one_dense_layer_call_and_return_conditional_losses_839283d8¢5
.¢+
)&
inputs’’’’’’’’’’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 
0__inference_width_one_dense_layer_call_fn_839290W8¢5
.¢+
)&
inputs’’’’’’’’’’’’’’’’’’
Ŗ "’’’’’’’’’