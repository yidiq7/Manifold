ящ
ЭЃ
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
О
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
 "serve*2.3.12v2.3.0-54-gfcc4b966f18Џ
m
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	с*
shared_name
Variable
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	с*
dtype0

NoOpNoOp
Г	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ю
valueфBс Bк
Љ
biholomorphic_k2

layer1

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
w
#	_self_saveable_object_factories

trainable_variables
regularization_losses
	variables
	keras_api
~
w
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
 
 

0
 

0
­
layer_metrics
trainable_variables
layer_regularization_losses
metrics
regularization_losses
	variables

layers
non_trainable_variables
 
 
 
 
­
layer_metrics

trainable_variables
layer_regularization_losses
metrics
regularization_losses
	variables

layers
non_trainable_variables
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 

0
 

0
­
layer_metrics
trainable_variables
layer_regularization_losses
 metrics
regularization_losses
	variables

!layers
"non_trainable_variables
 
 
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Р
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_20218
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
М
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_20244
Ѓ
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_20257ї
д
y
2__inference_outer_product_nn_k2_layer_call_fn_2144
input_1
unknown
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_21202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ў(
п	
K__forward_outer_product_nn_k2_layer_call_and_return_conditional_losses_2608
input_1
width_one_dense_8421146
identity+
'width_one_dense_statefulpartitionedcall-
)width_one_dense_statefulpartitionedcall_0-
)width_one_dense_statefulpartitionedcall_1$
 biholomorphic_k2_partitionedcall&
"biholomorphic_k2_partitionedcall_0&
"biholomorphic_k2_partitionedcall_1	&
"biholomorphic_k2_partitionedcall_2&
"biholomorphic_k2_partitionedcall_3&
"biholomorphic_k2_partitionedcall_4&
"biholomorphic_k2_partitionedcall_5&
"biholomorphic_k2_partitionedcall_6&
"biholomorphic_k2_partitionedcall_7&
"biholomorphic_k2_partitionedcall_8	&
"biholomorphic_k2_partitionedcall_9	'
#biholomorphic_k2_partitionedcall_10'
#biholomorphic_k2_partitionedcall_11'
#biholomorphic_k2_partitionedcall_12'
#biholomorphic_k2_partitionedcall_13'
#biholomorphic_k2_partitionedcall_14'
#biholomorphic_k2_partitionedcall_15	'
#biholomorphic_k2_partitionedcall_16'
#biholomorphic_k2_partitionedcall_17'
#biholomorphic_k2_partitionedcall_18'
#biholomorphic_k2_partitionedcall_19	'
#biholomorphic_k2_partitionedcall_20	'
#biholomorphic_k2_partitionedcall_21Ђ'width_one_dense/StatefulPartitionedCallШ
 biholomorphic_k2/PartitionedCallPartitionedCallinput_1*
Tin
2*$
Tout
2						*
_collective_manager_ids
 *ы
_output_shapesи
е:џџџџџџџџџџџџџџџџџџ::Тџџџџџџџџџ:џџџџџџџџџ:Тџџџџџџџџџ:: :џџџџџџџџџс:џџџџџџџџџс:џџџџџџџџџ: : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::џџџџџџџџџ: : :џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__forward_biholomorphic_k2_layer_call_and_return_conditional_losses_25572"
 biholomorphic_k2/PartitionedCallщ
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall)biholomorphic_k2/PartitionedCall:output:0width_one_dense_8421146*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::џџџџџџџџџ:	с:џџџџџџџџџџџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__forward_width_one_dense_layer_call_and_return_conditional_losses_24362)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Log
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"M
 biholomorphic_k2_partitionedcall)biholomorphic_k2/PartitionedCall:output:1"O
"biholomorphic_k2_partitionedcall_0)biholomorphic_k2/PartitionedCall:output:2"O
"biholomorphic_k2_partitionedcall_1)biholomorphic_k2/PartitionedCall:output:3"Q
#biholomorphic_k2_partitionedcall_10*biholomorphic_k2/PartitionedCall:output:12"Q
#biholomorphic_k2_partitionedcall_11*biholomorphic_k2/PartitionedCall:output:13"Q
#biholomorphic_k2_partitionedcall_12*biholomorphic_k2/PartitionedCall:output:14"Q
#biholomorphic_k2_partitionedcall_13*biholomorphic_k2/PartitionedCall:output:15"Q
#biholomorphic_k2_partitionedcall_14*biholomorphic_k2/PartitionedCall:output:16"Q
#biholomorphic_k2_partitionedcall_15*biholomorphic_k2/PartitionedCall:output:17"Q
#biholomorphic_k2_partitionedcall_16*biholomorphic_k2/PartitionedCall:output:18"Q
#biholomorphic_k2_partitionedcall_17*biholomorphic_k2/PartitionedCall:output:19"Q
#biholomorphic_k2_partitionedcall_18*biholomorphic_k2/PartitionedCall:output:20"Q
#biholomorphic_k2_partitionedcall_19*biholomorphic_k2/PartitionedCall:output:21"O
"biholomorphic_k2_partitionedcall_2)biholomorphic_k2/PartitionedCall:output:4"Q
#biholomorphic_k2_partitionedcall_20*biholomorphic_k2/PartitionedCall:output:22"Q
#biholomorphic_k2_partitionedcall_21*biholomorphic_k2/PartitionedCall:output:23"O
"biholomorphic_k2_partitionedcall_3)biholomorphic_k2/PartitionedCall:output:5"O
"biholomorphic_k2_partitionedcall_4)biholomorphic_k2/PartitionedCall:output:6"O
"biholomorphic_k2_partitionedcall_5)biholomorphic_k2/PartitionedCall:output:7"O
"biholomorphic_k2_partitionedcall_6)biholomorphic_k2/PartitionedCall:output:8"O
"biholomorphic_k2_partitionedcall_7)biholomorphic_k2/PartitionedCall:output:9"P
"biholomorphic_k2_partitionedcall_8*biholomorphic_k2/PartitionedCall:output:10"P
"biholomorphic_k2_partitionedcall_9*biholomorphic_k2/PartitionedCall:output:11"
identityIdentity:output:0"[
'width_one_dense_statefulpartitionedcall0width_one_dense/StatefulPartitionedCall:output:0"]
)width_one_dense_statefulpartitionedcall_00width_one_dense/StatefulPartitionedCall:output:1"]
)width_one_dense_statefulpartitionedcall_10width_one_dense/StatefulPartitionedCall:output:2**
_input_shapes
:џџџџџџџџџ:*y
backward_function_name_]__inference___backward_outer_product_nn_k2_layer_call_and_return_conditional_losses_2419_26092R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Љ

I__inference_width_one_dense_layer_call_and_return_conditional_losses_1757

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	с*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ::X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
K
/__inference_biholomorphic_k2_layer_call_fn_2112

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_21072
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Г
G__forward_width_one_dense_layer_call_and_return_conditional_losses_2436
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputs
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	с*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:*u
backward_function_name[Y__inference___backward_width_one_dense_layer_call_and_return_conditional_losses_2426_2437:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њ
n
'__inference_restored_function_body_2408
input_1
unknown
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_21202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
с|
Щ
H__forward_biholomorphic_k2_layer_call_and_return_conditional_losses_2557
inputs_0
identity
transpose_3_perm
boolean_mask_1_reshape
boolean_mask_1_squeeze	
transpose_2
transpose_2_perm
concat_axis
real
imag
matrixbandpart_1
matrixbandpart_1_num_lower	
matrixbandpart_1_num_upper	
	reshape_1
conj
transpose_1
transpose_1_perm
boolean_mask_reshape
boolean_mask_squeeze	
	transpose
transpose_perm
matrixbandpart
matrixbandpart_num_lower	
matrixbandpart_num_upper	

inputs
einsum/EinsumEinsuminputs_0inputs_0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart/num_upperЦ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permY
transpose_0	TransposeReshape:output:0transpose/perm:output:0*
T02
	transposeR
Abs
ComplexAbstranspose_0:y:0*'
_output_shapes
:џџџџџџџџџ2
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
Squeezeg
boolean_mask/ShapeShapetranspose_0:y:0*
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
#boolean_mask/Prod/reduction_indicesЂ
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
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
$boolean_mask/strided_slice_1/stack_2И
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1k
boolean_mask/Shape_2Shapetranspose_0:y:0*
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
$boolean_mask/strided_slice_2/stack_2И
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
boolean_mask/concat/axisќ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisћ
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permm
transpose_1_0	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T02
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape_1/shape
	Reshape_1Reshapetranspose_1_0:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
ConjЈ
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart_1/num_upperа
MatrixBandPart_1MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandPart_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџс   2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_1:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
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
:џџџџџџџџџТ2
concatu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm^
transpose_2_0	Transposeconcat:output:0transpose_2/perm:output:0*
T02
transpose_2[
Abs_1Abstranspose_2_0:y:0*
T0*(
_output_shapes
:Тџџџџџџџџџ2
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
:Т2
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
:Т2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:Т2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:Т2
	Squeeze_1m
boolean_mask_1/ShapeShapetranspose_2_0:y:0*
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
$boolean_mask_1/strided_slice/stack_2Ј
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
%boolean_mask_1/Prod/reduction_indicesЊ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodq
boolean_mask_1/Shape_1Shapetranspose_2_0:y:0*
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
&boolean_mask_1/strided_slice_1/stack_2Ф
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1q
boolean_mask_1/Shape_2Shapetranspose_2_0:y:0*
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
&boolean_mask_1/strided_slice_2/stack_2Ф
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
boolean_mask_1/concatЁ
boolean_mask_1/ReshapeReshapetranspose_2_0:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:Тџџџџџџџџџ2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
boolean_mask_1/Reshape_1/shapeЂ
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:Т2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџџџџџџџџџџ2
transpose_3l
IdentityIdentitytranspose_3:y:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"9
boolean_mask_1_reshapeboolean_mask_1/Reshape:output:0"9
boolean_mask_1_squeezeboolean_mask_1/Squeeze:output:0"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"'
matrixbandpartMatrixBandPart:band:0"+
matrixbandpart_1MatrixBandPart_1:band:0"A
matrixbandpart_1_num_lower#MatrixBandPart_1/num_lower:output:0"A
matrixbandpart_1_num_upper#MatrixBandPart_1/num_upper:output:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	reshape_1Reshape_1:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:џџџџџџџџџ*v
backward_function_name\Z__inference___backward_biholomorphic_k2_layer_call_and_return_conditional_losses_2444_2558:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


__inference__traced_save_20244
file_prefix'
#savev2_variable_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
value3B1 B+_temp_9f65436ffcfb43d4b57e86c1b5189dc4/part2	
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameУ
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
SaveV2/shape_and_slicesр
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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
: :	с: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	с:

_output_shapes
: 
Э

 __inference__wrapped_model_20209
input_1
outer_product_nn_k2_20205
identityЂ+outer_product_nn_k2/StatefulPartitionedCall
+outer_product_nn_k2/StatefulPartitionedCallStatefulPartitionedCallinput_1outer_product_nn_k2_20205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *0
f+R)
'__inference_restored_function_body_24082-
+outer_product_nn_k2/StatefulPartitionedCallЖ
IdentityIdentity4outer_product_nn_k2/StatefulPartitionedCall:output:0,^outer_product_nn_k2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2Z
+outer_product_nn_k2/StatefulPartitionedCall+outer_product_nn_k2/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ш

y
!__inference__traced_restore_20257
file_prefix
assignvariableop_variable

identity_2ЂAssignVariableOpЩ
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
RestoreV2/shape_and_slicesЕ
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
л
t
.__inference_width_one_dense_layer_call_fn_1876

inputs
unknown
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_width_one_dense_layer_call_and_return_conditional_losses_18702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
с|
Щ
H__forward_biholomorphic_k2_layer_call_and_return_conditional_losses_3409
inputs_0
identity
transpose_3_perm
boolean_mask_1_reshape
boolean_mask_1_squeeze	
transpose_2
transpose_2_perm
concat_axis
real
imag
matrixbandpart_1
matrixbandpart_1_num_lower	
matrixbandpart_1_num_upper	
	reshape_1
conj
transpose_1
transpose_1_perm
boolean_mask_reshape
boolean_mask_squeeze	
	transpose
transpose_perm
matrixbandpart
matrixbandpart_num_lower	
matrixbandpart_num_upper	

inputs
einsum/EinsumEinsuminputs_0inputs_0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart/num_upperЦ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permY
transpose_0	TransposeReshape:output:0transpose/perm:output:0*
T02
	transposeR
Abs
ComplexAbstranspose_0:y:0*'
_output_shapes
:џџџџџџџџџ2
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
Squeezeg
boolean_mask/ShapeShapetranspose_0:y:0*
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
#boolean_mask/Prod/reduction_indicesЂ
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
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
$boolean_mask/strided_slice_1/stack_2И
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1k
boolean_mask/Shape_2Shapetranspose_0:y:0*
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
$boolean_mask/strided_slice_2/stack_2И
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
boolean_mask/concat/axisќ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisћ
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permm
transpose_1_0	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T02
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape_1/shape
	Reshape_1Reshapetranspose_1_0:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
ConjЈ
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart_1/num_upperа
MatrixBandPart_1MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandPart_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџс   2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_1:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
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
:џџџџџџџџџТ2
concatu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm^
transpose_2_0	Transposeconcat:output:0transpose_2/perm:output:0*
T02
transpose_2[
Abs_1Abstranspose_2_0:y:0*
T0*(
_output_shapes
:Тџџџџџџџџџ2
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
:Т2
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
:Т2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:Т2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:Т2
	Squeeze_1m
boolean_mask_1/ShapeShapetranspose_2_0:y:0*
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
$boolean_mask_1/strided_slice/stack_2Ј
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
%boolean_mask_1/Prod/reduction_indicesЊ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodq
boolean_mask_1/Shape_1Shapetranspose_2_0:y:0*
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
&boolean_mask_1/strided_slice_1/stack_2Ф
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1q
boolean_mask_1/Shape_2Shapetranspose_2_0:y:0*
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
&boolean_mask_1/strided_slice_2/stack_2Ф
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
boolean_mask_1/concatЁ
boolean_mask_1/ReshapeReshapetranspose_2_0:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:Тџџџџџџџџџ2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
boolean_mask_1/Reshape_1/shapeЂ
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:Т2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџџџџџџџџџџ2
transpose_3l
IdentityIdentitytranspose_3:y:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"9
boolean_mask_1_reshapeboolean_mask_1/Reshape:output:0"9
boolean_mask_1_squeezeboolean_mask_1/Squeeze:output:0"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"'
matrixbandpartMatrixBandPart:band:0"+
matrixbandpart_1MatrixBandPart_1:band:0"A
matrixbandpart_1_num_lower#MatrixBandPart_1/num_lower:output:0"A
matrixbandpart_1_num_upper#MatrixBandPart_1/num_upper:output:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	reshape_1Reshape_1:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:џџџџџџџџџ*v
backward_function_name\Z__inference___backward_biholomorphic_k2_layer_call_and_return_conditional_losses_3262_3410:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў(
п	
K__forward_outer_product_nn_k2_layer_call_and_return_conditional_losses_3437
input_1
width_one_dense_8421146
identity+
'width_one_dense_statefulpartitionedcall-
)width_one_dense_statefulpartitionedcall_0-
)width_one_dense_statefulpartitionedcall_1$
 biholomorphic_k2_partitionedcall&
"biholomorphic_k2_partitionedcall_0&
"biholomorphic_k2_partitionedcall_1	&
"biholomorphic_k2_partitionedcall_2&
"biholomorphic_k2_partitionedcall_3&
"biholomorphic_k2_partitionedcall_4&
"biholomorphic_k2_partitionedcall_5&
"biholomorphic_k2_partitionedcall_6&
"biholomorphic_k2_partitionedcall_7&
"biholomorphic_k2_partitionedcall_8	&
"biholomorphic_k2_partitionedcall_9	'
#biholomorphic_k2_partitionedcall_10'
#biholomorphic_k2_partitionedcall_11'
#biholomorphic_k2_partitionedcall_12'
#biholomorphic_k2_partitionedcall_13'
#biholomorphic_k2_partitionedcall_14'
#biholomorphic_k2_partitionedcall_15	'
#biholomorphic_k2_partitionedcall_16'
#biholomorphic_k2_partitionedcall_17'
#biholomorphic_k2_partitionedcall_18'
#biholomorphic_k2_partitionedcall_19	'
#biholomorphic_k2_partitionedcall_20	'
#biholomorphic_k2_partitionedcall_21Ђ'width_one_dense/StatefulPartitionedCallШ
 biholomorphic_k2/PartitionedCallPartitionedCallinput_1*
Tin
2*$
Tout
2						*
_collective_manager_ids
 *ы
_output_shapesи
е:џџџџџџџџџџџџџџџџџџ::Тџџџџџџџџџ:џџџџџџџџџ:Тџџџџџџџџџ:: :џџџџџџџџџс:џџџџџџџџџс:џџџџџџџџџ: : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::џџџџџџџџџ: : :џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__forward_biholomorphic_k2_layer_call_and_return_conditional_losses_34092"
 biholomorphic_k2/PartitionedCallщ
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall)biholomorphic_k2/PartitionedCall:output:0width_one_dense_8421146*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::џџџџџџџџџ:	с:џџџџџџџџџџџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__forward_width_one_dense_layer_call_and_return_conditional_losses_32562)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Log
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"M
 biholomorphic_k2_partitionedcall)biholomorphic_k2/PartitionedCall:output:1"O
"biholomorphic_k2_partitionedcall_0)biholomorphic_k2/PartitionedCall:output:2"O
"biholomorphic_k2_partitionedcall_1)biholomorphic_k2/PartitionedCall:output:3"Q
#biholomorphic_k2_partitionedcall_10*biholomorphic_k2/PartitionedCall:output:12"Q
#biholomorphic_k2_partitionedcall_11*biholomorphic_k2/PartitionedCall:output:13"Q
#biholomorphic_k2_partitionedcall_12*biholomorphic_k2/PartitionedCall:output:14"Q
#biholomorphic_k2_partitionedcall_13*biholomorphic_k2/PartitionedCall:output:15"Q
#biholomorphic_k2_partitionedcall_14*biholomorphic_k2/PartitionedCall:output:16"Q
#biholomorphic_k2_partitionedcall_15*biholomorphic_k2/PartitionedCall:output:17"Q
#biholomorphic_k2_partitionedcall_16*biholomorphic_k2/PartitionedCall:output:18"Q
#biholomorphic_k2_partitionedcall_17*biholomorphic_k2/PartitionedCall:output:19"Q
#biholomorphic_k2_partitionedcall_18*biholomorphic_k2/PartitionedCall:output:20"Q
#biholomorphic_k2_partitionedcall_19*biholomorphic_k2/PartitionedCall:output:21"O
"biholomorphic_k2_partitionedcall_2)biholomorphic_k2/PartitionedCall:output:4"Q
#biholomorphic_k2_partitionedcall_20*biholomorphic_k2/PartitionedCall:output:22"Q
#biholomorphic_k2_partitionedcall_21*biholomorphic_k2/PartitionedCall:output:23"O
"biholomorphic_k2_partitionedcall_3)biholomorphic_k2/PartitionedCall:output:5"O
"biholomorphic_k2_partitionedcall_4)biholomorphic_k2/PartitionedCall:output:6"O
"biholomorphic_k2_partitionedcall_5)biholomorphic_k2/PartitionedCall:output:7"O
"biholomorphic_k2_partitionedcall_6)biholomorphic_k2/PartitionedCall:output:8"O
"biholomorphic_k2_partitionedcall_7)biholomorphic_k2/PartitionedCall:output:9"P
"biholomorphic_k2_partitionedcall_8*biholomorphic_k2/PartitionedCall:output:10"P
"biholomorphic_k2_partitionedcall_9*biholomorphic_k2/PartitionedCall:output:11"
identityIdentity:output:0"[
'width_one_dense_statefulpartitionedcall0width_one_dense/StatefulPartitionedCall:output:0"]
)width_one_dense_statefulpartitionedcall_00width_one_dense/StatefulPartitionedCall:output:1"]
)width_one_dense_statefulpartitionedcall_10width_one_dense/StatefulPartitionedCall:output:2**
_input_shapes
:џџџџџџџџџ:*y
backward_function_name_]__inference___backward_outer_product_nn_k2_layer_call_and_return_conditional_losses_3202_34382R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ђ
Д
M__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_2120
input_1
width_one_dense_8421146
identityЂ'width_one_dense/StatefulPartitionedCallГ
 biholomorphic_k2/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *ы
_output_shapesи
е:џџџџџџџџџџџџџџџџџџ::Тџџџџџџџџџ:џџџџџџџџџ:Тџџџџџџџџџ:: :џџџџџџџџџс:џџџџџџџџџс:џџџџџџџџџ: : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::џџџџџџџџџ: : :џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_21072"
 biholomorphic_k2/PartitionedCallТ
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall)biholomorphic_k2/PartitionedCall:output:0width_one_dense_8421146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_width_one_dense_layer_call_and_return_conditional_losses_18702)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Log
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1

Г
G__forward_width_one_dense_layer_call_and_return_conditional_losses_3256
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputs
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	с*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:*u
backward_function_name[Y__inference___backward_width_one_dense_layer_call_and_return_conditional_losses_3240_3257:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Љ

I__inference_width_one_dense_layer_call_and_return_conditional_losses_1870

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	с*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ::X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

j
#__inference_signature_wrapper_20218
input_1
unknown
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_202092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ѕp
f
J__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_1856

inputs
identity
einsum/EinsumEinsuminputsinputs*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart/num_upperЦ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
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
:џџџџџџџџџ2
	transposeP
Abs
ComplexAbstranspose:y:0*'
_output_shapes
:џџџџџџџџџ2
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
#boolean_mask/Prod/reduction_indicesЂ
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
$boolean_mask/strided_slice_1/stack_2И
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
$boolean_mask/strided_slice_2/stack_2И
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
boolean_mask/concat/axisќ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisћ
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџџџџџџџџџџ2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape_1/shape~
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
ConjЈ
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart_1/num_upperа
MatrixBandPart_1MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandPart_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџс   2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_1:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
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
:џџџџџџџџџТ2
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
:Тџџџџџџџџџ2
transpose_2Y
Abs_1Abstranspose_2:y:0*
T0*(
_output_shapes
:Тџџџџџџџџџ2
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
:Т2
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
:Т2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:Т2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:Т2
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
$boolean_mask_1/strided_slice/stack_2Ј
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
%boolean_mask_1/Prod/reduction_indicesЊ
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
&boolean_mask_1/strided_slice_1/stack_2Ф
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
&boolean_mask_1/strided_slice_2/stack_2Ф
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
:Тџџџџџџџџџ2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
boolean_mask_1/Reshape_1/shapeЂ
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:Т2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџџџџџџџџџџ2
transpose_3l
IdentityIdentitytranspose_3:y:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѕp
f
J__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_2107

inputs
identity
einsum/EinsumEinsuminputsinputs*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart/num_upperЦ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
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
:џџџџџџџџџ2
	transposeP
Abs
ComplexAbstranspose:y:0*'
_output_shapes
:џџџџџџџџџ2
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
#boolean_mask/Prod/reduction_indicesЂ
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
$boolean_mask/strided_slice_1/stack_2И
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
$boolean_mask/strided_slice_2/stack_2И
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
boolean_mask/concat/axisќ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisћ
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџџџџџџџџџџ2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape_1/shape~
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
ConjЈ
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
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
џџџџџџџџџ2
MatrixBandPart_1/num_upperа
MatrixBandPart_1MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandPart_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџс   2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_1:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:џџџџџџџџџс2
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
:џџџџџџџџџТ2
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
:Тџџџџџџџџџ2
transpose_2Y
Abs_1Abstranspose_2:y:0*
T0*(
_output_shapes
:Тџџџџџџџџџ2
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
:Т2
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
:Т2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:Т2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:Т2
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
$boolean_mask_1/strided_slice/stack_2Ј
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
%boolean_mask_1/Prod/reduction_indicesЊ
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
&boolean_mask_1/strided_slice_1/stack_2Ф
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
&boolean_mask_1/strided_slice_2/stack_2Ф
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
:Тџџџџџџџџџ2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2 
boolean_mask_1/Reshape_1/shapeЂ
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:Т2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџџџџџџџџџџ2
transpose_3l
IdentityIdentitytranspose_3:y:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Љ3

biholomorphic_k2

layer1

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
#_default_save_signature
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_model§{"class_name": "OuterProductNN_k2", "name": "outer_product_nn_k2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "OuterProductNN_k2"}}
ђ
#	_self_saveable_object_factories

trainable_variables
regularization_losses
	variables
	keras_api
&__call__
*'&call_and_return_all_conditional_losses"О
_tf_keras_layerЄ{"class_name": "Biholomorphic_k2", "name": "biholomorphic_k2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic_k2", "trainable": true, "dtype": "float32"}}
к
w
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "WidthOneDense", "name": "width_one_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
,
*serving_default"
signature_map
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
Ъ
layer_metrics
trainable_variables
layer_regularization_losses
metrics
regularization_losses
	variables

layers
non_trainable_variables
$__call__
#_default_save_signature
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
layer_metrics

trainable_variables
layer_regularization_losses
metrics
regularization_losses
	variables

layers
non_trainable_variables
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
:	с2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
layer_metrics
trainable_variables
layer_regularization_losses
 metrics
regularization_losses
	variables

!layers
"non_trainable_variables
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
о2л
 __inference__wrapped_model_20209Ж
В
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
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
і2ѓ
2__inference_outer_product_nn_k2_layer_call_fn_2144М
В
FullArgSpec
args

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
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
2
M__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_2120М
В
FullArgSpec
args

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
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
Я2Ь
/__inference_biholomorphic_k2_layer_call_fn_2112
В
FullArgSpec
args

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
annotationsЊ *
 
ъ2ч
J__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_1856
В
FullArgSpec
args

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
annotationsЊ *
 
Ю2Ы
.__inference_width_one_dense_layer_call_fn_1876
В
FullArgSpec
args

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
annotationsЊ *
 
щ2ц
I__inference_width_one_dense_layer_call_and_return_conditional_losses_1757
В
FullArgSpec
args

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
annotationsЊ *
 
2B0
#__inference_signature_wrapper_20218input_1
 __inference__wrapped_model_20209j0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџЏ
J__inference_biholomorphic_k2_layer_call_and_return_conditional_losses_1856a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 
/__inference_biholomorphic_k2_layer_call_fn_2112T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџ­
M__inference_outer_product_nn_k2_layer_call_and_return_conditional_losses_2120\0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
2__inference_outer_product_nn_k2_layer_call_fn_2144O0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "џџџџџџџџџ
#__inference_signature_wrapper_20218u;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџБ
I__inference_width_one_dense_layer_call_and_return_conditional_losses_1757d8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_width_one_dense_layer_call_fn_1876W8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "џџџџџџџџџ