шн
═Б
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.12v2.3.0-54-gfcc4b966f18Џ№
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:F*
dtype0
p

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:FF*
shared_name
Variable_1
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:FF*
dtype0
p

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:FF*
shared_name
Variable_2
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:FF*
dtype0
p

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*
shared_name
Variable_3
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes

:F*
dtype0

NoOpNoOp
м
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ї
valueЃBђ Bщ
╩
biholomorphic

layer1

layer2

layer3

layer4

signatures
#_self_saveable_object_factories
regularization_losses
	trainable_variables

	variables
	keras_api
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
~
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
~
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
~
w
#_self_saveable_object_factories
regularization_losses
 trainable_variables
!	variables
"	keras_api
~
#w
#$_self_saveable_object_factories
%regularization_losses
&trainable_variables
'	variables
(	keras_api
 
 
 

0
1
2
#3

0
1
2
#3
Г
regularization_losses

)layers
	trainable_variables
*non_trainable_variables
+layer_regularization_losses
,metrics

	variables
-layer_metrics
 
 
 
 
Г
regularization_losses

.layers
trainable_variables
/non_trainable_variables
0layer_regularization_losses
1metrics
	variables
2layer_metrics
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
Г
regularization_losses

3layers
trainable_variables
4non_trainable_variables
5layer_regularization_losses
6metrics
	variables
7layer_metrics
CA
VARIABLE_VALUE
Variable_1#layer2/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
Г
regularization_losses

8layers
trainable_variables
9non_trainable_variables
:layer_regularization_losses
;metrics
	variables
<layer_metrics
CA
VARIABLE_VALUE
Variable_2#layer3/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
Г
regularization_losses

=layers
 trainable_variables
>non_trainable_variables
?layer_regularization_losses
@metrics
!	variables
Alayer_metrics
CA
VARIABLE_VALUE
Variable_3#layer4/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

#0

#0
Г
%regularization_losses

Blayers
&trainable_variables
Cnon_trainable_variables
Dlayer_regularization_losses
Emetrics
'	variables
Flayer_metrics
#
0
1
2
3
4
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
 
z
serving_default_input_1Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
ь
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable
Variable_1
Variable_2
Variable_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ */
f*R(
&__inference_signature_wrapper_25052971
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
б
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpConst*
Tin

2*
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
GPU2*0J 8ѓ **
f%R#
!__inference__traced_save_25053006
═
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2
Variable_3*
Tin	
2*
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
GPU2*0J 8ѓ *-
f(R&
$__inference__traced_restore_25053028Ј╩
╦9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1983

inputs
identityE
ConjConjinputs*'
_output_shapes
:         2
Conjў
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upperк
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisІ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transposeconcat:output:0transpose/perm:output:0*
T0*'
_output_shapes
:2         2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2         2
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
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shapeј
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackњ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1њ
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2ю
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceћ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesб
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1њ
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackќ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1ќ
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
T0*
_output_shapes
:2
boolean_mask/Shape_2њ
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackќ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1ќ
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
boolean_mask/strided_slice_2ј
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
boolean_mask/concat/axisЧ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatќ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeЈ
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЎ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/Whereў
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisч
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЮ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З
и
?__forward_dense_2_layer_call_and_return_conditional_losses_3297
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         F:*m
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_3274_3298:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
╣
l
&__inference_dense_1_layer_call_fn_1799

inputs
unknown
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_17932
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0**
_input_shapes
:         F:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
Г
H
,__inference_biholomorphic_layer_call_fn_2011

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_biholomorphic_layer_call_and_return_conditional_losses_19832
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З
и
?__forward_dense_1_layer_call_and_return_conditional_losses_2298
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         F:*m
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_2284_2299:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
А
ў
!__inference__traced_save_25053006
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_95811d445f8c4dcb88b0d2f713da7925/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameх
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*К
valueйB║B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesњ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesп
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*?
_input_shapes.
,: :F:FF:FF:F: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:F:$ 

_output_shapes

:FF:$ 

_output_shapes

:FF:$ 

_output_shapes

:F:

_output_shapes
: 
ѕ
Џ
'__inference_restored_function_body_2218
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_kahler_potential_layer_call_and_return_conditional_losses_19972
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
Ф@
█
E__forward_biholomorphic_layer_call_and_return_conditional_losses_2399
inputs_0
identity
transpose_1_perm
boolean_mask_reshape
boolean_mask_squeeze	
	transpose
transpose_perm
concat_axis
real
imag
matrixbandpart
matrixbandpart_num_lower	
matrixbandpart_num_upper	

inputs
conjG
ConjConjinputs_0*'
_output_shapes
:         2
Conjџ
einsum/EinsumEinsuminputs_0Conj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upperк
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisІ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permX
transpose_0	Transposeconcat:output:0transpose/perm:output:0*
T02
	transposeT
AbsAbstranspose_0:y:0*
T0*'
_output_shapes
:2         2
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
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezeg
boolean_mask/ShapeShapetranspose_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shapeј
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackњ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1њ
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2ю
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceћ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesб
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1њ
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackќ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1ќ
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
T0*
_output_shapes
:2
boolean_mask/Shape_2њ
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackќ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1ќ
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
boolean_mask/strided_slice_2ј
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
boolean_mask/concat/axisЧ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatў
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeЈ
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЎ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/Whereў
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisч
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЮ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

Identity"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"'
matrixbandpartMatrixBandPart:band:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:         *s
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_2332_2400:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
К
j
$__inference_dense_layer_call_fn_1834

inputs
unknown
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_18282
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :                  :22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
 
џ
&__inference_signature_wrapper_25052971
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference__wrapped_model_250529562
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
У
ё
A__inference_dense_1_layer_call_and_return_conditional_losses_1842

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0**
_input_shapes
:         F::O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
Э
ѓ
?__inference_dense_layer_call_and_return_conditional_losses_1828

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :                  ::X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
ј
ё
A__inference_dense_3_layer_call_and_return_conditional_losses_1779

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         F::O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
Ы
Ф
?__forward_dense_3_layer_call_and_return_conditional_losses_3268
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         F:*m
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_3252_3269:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
Э
ѓ
?__inference_dense_layer_call_and_return_conditional_losses_1772

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :                  ::X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
ј
ё
A__inference_dense_3_layer_call_and_return_conditional_losses_1849

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         F::O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
З
и
?__forward_dense_2_layer_call_and_return_conditional_losses_2274
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         F:*m
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_2260_2275:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
ѓ
х
=__forward_dense_layer_call_and_return_conditional_losses_2322
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :                  :*k
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_2308_2323:X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
У
ё
A__inference_dense_2_layer_call_and_return_conditional_losses_1758

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0**
_input_shapes
:         F::O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
╦2
в	
H__forward_kahler_potential_layer_call_and_return_conditional_losses_2433
input_1
dense_66071450
dense_1_66071470
dense_2_66071490
dense_3_66071509
identity#
dense_3_statefulpartitionedcall%
!dense_3_statefulpartitionedcall_0%
!dense_3_statefulpartitionedcall_1#
dense_2_statefulpartitionedcall%
!dense_2_statefulpartitionedcall_0%
!dense_2_statefulpartitionedcall_1#
dense_1_statefulpartitionedcall%
!dense_1_statefulpartitionedcall_0%
!dense_1_statefulpartitionedcall_1!
dense_statefulpartitionedcall#
dense_statefulpartitionedcall_0#
dense_statefulpartitionedcall_1!
biholomorphic_partitionedcall#
biholomorphic_partitionedcall_0#
biholomorphic_partitionedcall_1	#
biholomorphic_partitionedcall_2#
biholomorphic_partitionedcall_3#
biholomorphic_partitionedcall_4#
biholomorphic_partitionedcall_5#
biholomorphic_partitionedcall_6#
biholomorphic_partitionedcall_7#
biholomorphic_partitionedcall_8	#
biholomorphic_partitionedcall_9	$
 biholomorphic_partitionedcall_10$
 biholomorphic_partitionedcall_11ѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallд
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2			*
_collective_manager_ids
 *▄
_output_shapes╔
к:                  ::2         :         :2         :: :         :         :         : : :         :         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__forward_biholomorphic_layer_call_and_return_conditional_losses_23992
biholomorphic/PartitionedCallм
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_66071450*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         F:         F:F:                  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *F
fAR?
=__forward_dense_layer_call_and_return_conditional_losses_23222
dense/StatefulPartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_66071470*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         F:         F:FF:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__forward_dense_1_layer_call_and_return_conditional_losses_22982!
dense_1/StatefulPartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_66071490*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         F:         F:FF:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__forward_dense_2_layer_call_and_return_conditional_losses_22742!
dense_2/StatefulPartitionedCall┐
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_66071509*
Tin
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:         :F:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__forward_dense_3_layer_call_and_return_conditional_losses_22522!
dense_3/StatefulPartitionedCallm
LogLog(dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2
Logр
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"G
biholomorphic_partitionedcall&biholomorphic/PartitionedCall:output:1"I
biholomorphic_partitionedcall_0&biholomorphic/PartitionedCall:output:2"I
biholomorphic_partitionedcall_1&biholomorphic/PartitionedCall:output:3"K
 biholomorphic_partitionedcall_10'biholomorphic/PartitionedCall:output:12"K
 biholomorphic_partitionedcall_11'biholomorphic/PartitionedCall:output:13"I
biholomorphic_partitionedcall_2&biholomorphic/PartitionedCall:output:4"I
biholomorphic_partitionedcall_3&biholomorphic/PartitionedCall:output:5"I
biholomorphic_partitionedcall_4&biholomorphic/PartitionedCall:output:6"I
biholomorphic_partitionedcall_5&biholomorphic/PartitionedCall:output:7"I
biholomorphic_partitionedcall_6&biholomorphic/PartitionedCall:output:8"I
biholomorphic_partitionedcall_7&biholomorphic/PartitionedCall:output:9"J
biholomorphic_partitionedcall_8'biholomorphic/PartitionedCall:output:10"J
biholomorphic_partitionedcall_9'biholomorphic/PartitionedCall:output:11"K
dense_1_statefulpartitionedcall(dense_1/StatefulPartitionedCall:output:1"M
!dense_1_statefulpartitionedcall_0(dense_1/StatefulPartitionedCall:output:2"M
!dense_1_statefulpartitionedcall_1(dense_1/StatefulPartitionedCall:output:3"K
dense_2_statefulpartitionedcall(dense_2/StatefulPartitionedCall:output:1"M
!dense_2_statefulpartitionedcall_0(dense_2/StatefulPartitionedCall:output:2"M
!dense_2_statefulpartitionedcall_1(dense_2/StatefulPartitionedCall:output:3"K
dense_3_statefulpartitionedcall(dense_3/StatefulPartitionedCall:output:0"M
!dense_3_statefulpartitionedcall_0(dense_3/StatefulPartitionedCall:output:1"M
!dense_3_statefulpartitionedcall_1(dense_3/StatefulPartitionedCall:output:2"G
dense_statefulpartitionedcall&dense/StatefulPartitionedCall:output:1"I
dense_statefulpartitionedcall_0&dense/StatefulPartitionedCall:output:2"I
dense_statefulpartitionedcall_1&dense/StatefulPartitionedCall:output:3"
identityIdentity:output:0*6
_input_shapes%
#:         ::::*v
backward_function_name\Z__inference___backward_kahler_potential_layer_call_and_return_conditional_losses_2235_24342>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
ч
Ж
#__inference__wrapped_model_25052956
input_1
kahler_potential_25052946
kahler_potential_25052948
kahler_potential_25052950
kahler_potential_25052952
identityѕб(kahler_potential/StatefulPartitionedCall┘
(kahler_potential/StatefulPartitionedCallStatefulPartitionedCallinput_1kahler_potential_25052946kahler_potential_25052948kahler_potential_25052950kahler_potential_25052952*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *0
f+R)
'__inference_restored_function_body_22182*
(kahler_potential/StatefulPartitionedCall░
IdentityIdentity1kahler_potential/StatefulPartitionedCall:output:0)^kahler_potential/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2T
(kahler_potential/StatefulPartitionedCall(kahler_potential/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
З
и
?__forward_dense_1_layer_call_and_return_conditional_losses_3327
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         F:*m
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_3304_3328:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
э
к
J__inference_kahler_potential_layer_call_and_return_conditional_losses_1997
input_1
dense_66071450
dense_1_66071470
dense_2_66071490
dense_3_66071509
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallЏ
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *▄
_output_shapes╔
к:                  ::2         :         :2         :: :         :         :         : : :         :         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_biholomorphic_layer_call_and_return_conditional_losses_19832
biholomorphic/PartitionedCallў
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_66071450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_18282
dense/StatefulPartitionedCallа
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_66071470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_17932!
dense_1/StatefulPartitionedCallб
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_66071490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_17582!
dense_2/StatefulPartitionedCallб
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_66071509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_17792!
dense_3/StatefulPartitionedCallm
LogLog(dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2
Logр
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╦2
в	
H__forward_kahler_potential_layer_call_and_return_conditional_losses_3472
input_1
dense_66071450
dense_1_66071470
dense_2_66071490
dense_3_66071509
identity#
dense_3_statefulpartitionedcall%
!dense_3_statefulpartitionedcall_0%
!dense_3_statefulpartitionedcall_1#
dense_2_statefulpartitionedcall%
!dense_2_statefulpartitionedcall_0%
!dense_2_statefulpartitionedcall_1#
dense_1_statefulpartitionedcall%
!dense_1_statefulpartitionedcall_0%
!dense_1_statefulpartitionedcall_1!
dense_statefulpartitionedcall#
dense_statefulpartitionedcall_0#
dense_statefulpartitionedcall_1!
biholomorphic_partitionedcall#
biholomorphic_partitionedcall_0#
biholomorphic_partitionedcall_1	#
biholomorphic_partitionedcall_2#
biholomorphic_partitionedcall_3#
biholomorphic_partitionedcall_4#
biholomorphic_partitionedcall_5#
biholomorphic_partitionedcall_6#
biholomorphic_partitionedcall_7#
biholomorphic_partitionedcall_8	#
biholomorphic_partitionedcall_9	$
 biholomorphic_partitionedcall_10$
 biholomorphic_partitionedcall_11ѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallд
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2			*
_collective_manager_ids
 *▄
_output_shapes╔
к:                  ::2         :         :2         :: :         :         :         : : :         :         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__forward_biholomorphic_layer_call_and_return_conditional_losses_34512
biholomorphic/PartitionedCallм
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_66071450*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         F:         F:F:                  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *F
fAR?
=__forward_dense_layer_call_and_return_conditional_losses_33572
dense/StatefulPartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_66071470*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         F:         F:FF:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__forward_dense_1_layer_call_and_return_conditional_losses_33272!
dense_1/StatefulPartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_66071490*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         F:         F:FF:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__forward_dense_2_layer_call_and_return_conditional_losses_32972!
dense_2/StatefulPartitionedCall┐
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_66071509*
Tin
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:         :F:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__forward_dense_3_layer_call_and_return_conditional_losses_32682!
dense_3/StatefulPartitionedCallm
LogLog(dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2
Logр
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"G
biholomorphic_partitionedcall&biholomorphic/PartitionedCall:output:1"I
biholomorphic_partitionedcall_0&biholomorphic/PartitionedCall:output:2"I
biholomorphic_partitionedcall_1&biholomorphic/PartitionedCall:output:3"K
 biholomorphic_partitionedcall_10'biholomorphic/PartitionedCall:output:12"K
 biholomorphic_partitionedcall_11'biholomorphic/PartitionedCall:output:13"I
biholomorphic_partitionedcall_2&biholomorphic/PartitionedCall:output:4"I
biholomorphic_partitionedcall_3&biholomorphic/PartitionedCall:output:5"I
biholomorphic_partitionedcall_4&biholomorphic/PartitionedCall:output:6"I
biholomorphic_partitionedcall_5&biholomorphic/PartitionedCall:output:7"I
biholomorphic_partitionedcall_6&biholomorphic/PartitionedCall:output:8"I
biholomorphic_partitionedcall_7&biholomorphic/PartitionedCall:output:9"J
biholomorphic_partitionedcall_8'biholomorphic/PartitionedCall:output:10"J
biholomorphic_partitionedcall_9'biholomorphic/PartitionedCall:output:11"K
dense_1_statefulpartitionedcall(dense_1/StatefulPartitionedCall:output:1"M
!dense_1_statefulpartitionedcall_0(dense_1/StatefulPartitionedCall:output:2"M
!dense_1_statefulpartitionedcall_1(dense_1/StatefulPartitionedCall:output:3"K
dense_2_statefulpartitionedcall(dense_2/StatefulPartitionedCall:output:1"M
!dense_2_statefulpartitionedcall_0(dense_2/StatefulPartitionedCall:output:2"M
!dense_2_statefulpartitionedcall_1(dense_2/StatefulPartitionedCall:output:3"K
dense_3_statefulpartitionedcall(dense_3/StatefulPartitionedCall:output:0"M
!dense_3_statefulpartitionedcall_0(dense_3/StatefulPartitionedCall:output:1"M
!dense_3_statefulpartitionedcall_1(dense_3/StatefulPartitionedCall:output:2"G
dense_statefulpartitionedcall&dense/StatefulPartitionedCall:output:1"I
dense_statefulpartitionedcall_0&dense/StatefulPartitionedCall:output:2"I
dense_statefulpartitionedcall_1&dense/StatefulPartitionedCall:output:3"
identityIdentity:output:0*6
_input_shapes%
#:         ::::*v
backward_function_name\Z__inference___backward_kahler_potential_layer_call_and_return_conditional_losses_3206_34732>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╣
l
&__inference_dense_3_layer_call_fn_1785

inputs
unknown
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_17792
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         F:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
»
Б
/__inference_kahler_potential_layer_call_fn_2006
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_kahler_potential_layer_call_and_return_conditional_losses_19972
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
У
ё
A__inference_dense_1_layer_call_and_return_conditional_losses_1793

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0**
_input_shapes
:         F::O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
╦9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1922

inputs
identityE
ConjConjinputs*'
_output_shapes
:         2
Conjў
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upperк
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisІ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transposeconcat:output:0transpose/perm:output:0*
T0*'
_output_shapes
:2         2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2         2
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
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shapeј
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackњ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1њ
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2ю
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceћ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesб
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1њ
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackќ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1ќ
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
T0*
_output_shapes
:2
boolean_mask/Shape_2њ
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackќ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1ќ
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
boolean_mask/strided_slice_2ј
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
boolean_mask/concat/axisЧ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatќ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeЈ
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЎ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/Whereў
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisч
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЮ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
█
ц
$__inference__traced_restore_25053028
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3

identity_5ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3╗
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*К
valueйB║B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesў
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices─
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityў
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1б
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2б
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3б
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4г

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ф@
█
E__forward_biholomorphic_layer_call_and_return_conditional_losses_3451
inputs_0
identity
transpose_1_perm
boolean_mask_reshape
boolean_mask_squeeze	
	transpose
transpose_perm
concat_axis
real
imag
matrixbandpart
matrixbandpart_num_lower	
matrixbandpart_num_upper	

inputs
conjG
ConjConjinputs_0*'
_output_shapes
:         2
Conjџ
einsum/EinsumEinsuminputs_0Conj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upperк
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisІ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permX
transpose_0	Transposeconcat:output:0transpose/perm:output:0*
T02
	transposeT
AbsAbstranspose_0:y:0*
T0*'
_output_shapes
:2         2
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
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezeg
boolean_mask/ShapeShapetranspose_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shapeј
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackњ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1њ
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2ю
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceћ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesб
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1њ
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackќ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1ќ
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
T0*
_output_shapes
:2
boolean_mask/Shape_2њ
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackќ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1ќ
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
boolean_mask/strided_slice_2ј
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
boolean_mask/concat/axisЧ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatў
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeЈ
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЎ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/Whereў
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisч
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЮ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

Identity"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"'
matrixbandpartMatrixBandPart:band:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:         *s
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_3364_3452:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы
Ф
?__forward_dense_3_layer_call_and_return_conditional_losses_2252
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         F:*m
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_2242_2253:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
ѓ
х
=__forward_dense_layer_call_and_return_conditional_losses_3357
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :                  :*k
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_3334_3358:X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
╣
l
&__inference_dense_2_layer_call_fn_1764

inputs
unknown
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         F*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_17582
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0**
_input_shapes
:         F:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         F
 
_user_specified_nameinputs
У
ё
A__inference_dense_2_layer_call_and_return_conditional_losses_1930

inputs"
matmul_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:FF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         F2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         F2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0**
_input_shapes
:         F::O K
'
_output_shapes
:         F
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ф
serving_defaultЌ
;
input_10
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:д^
┤
biholomorphic

layer1

layer2

layer3

layer4

signatures
#_self_saveable_object_factories
regularization_losses
	trainable_variables

	variables
	keras_api
G_default_save_signature
*H&call_and_return_all_conditional_losses
I__call__"љ
_tf_keras_modelШ{"class_name": "KahlerPotential", "name": "kahler_potential", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "KahlerPotential"}}
ж
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"х
_tf_keras_layerЏ{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
╚
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"Ї
_tf_keras_layerз{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
╩
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*N&call_and_return_all_conditional_losses
O__call__"Ј
_tf_keras_layerш{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
╩
w
#_self_saveable_object_factories
regularization_losses
 trainable_variables
!	variables
"	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"Ј
_tf_keras_layerш{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
╩
#w
#$_self_saveable_object_factories
%regularization_losses
&trainable_variables
'	variables
(	keras_api
*R&call_and_return_all_conditional_losses
S__call__"Ј
_tf_keras_layerш{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
,
Tserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
#3"
trackable_list_wrapper
<
0
1
2
#3"
trackable_list_wrapper
╩
regularization_losses

)layers
	trainable_variables
*non_trainable_variables
+layer_regularization_losses
,metrics

	variables
-layer_metrics
I__call__
G_default_save_signature
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
regularization_losses

.layers
trainable_variables
/non_trainable_variables
0layer_regularization_losses
1metrics
	variables
2layer_metrics
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
:F2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
Г
regularization_losses

3layers
trainable_variables
4non_trainable_variables
5layer_regularization_losses
6metrics
	variables
7layer_metrics
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
:FF2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
Г
regularization_losses

8layers
trainable_variables
9non_trainable_variables
:layer_regularization_losses
;metrics
	variables
<layer_metrics
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
:FF2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
Г
regularization_losses

=layers
 trainable_variables
>non_trainable_variables
?layer_regularization_losses
@metrics
!	variables
Alayer_metrics
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
:F2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
Г
%regularization_losses

Blayers
&trainable_variables
Cnon_trainable_variables
Dlayer_regularization_losses
Emetrics
'	variables
Flayer_metrics
S__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
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
р2я
#__inference__wrapped_model_25052956Х
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         
ј2І
J__inference_kahler_potential_layer_call_and_return_conditional_losses_1997╝
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         
з2­
/__inference_kahler_potential_layer_call_fn_2006╝
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         
у2С
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1922ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠2╔
,__inference_biholomorphic_layer_call_fn_2011ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
?__inference_dense_layer_call_and_return_conditional_losses_1772ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
─2┴
$__inference_dense_layer_call_fn_1834ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
р2я
A__inference_dense_1_layer_call_and_return_conditional_losses_1842ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
&__inference_dense_1_layer_call_fn_1799ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
р2я
A__inference_dense_2_layer_call_and_return_conditional_losses_1930ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
&__inference_dense_2_layer_call_fn_1764ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
р2я
A__inference_dense_3_layer_call_and_return_conditional_losses_1849ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
&__inference_dense_3_layer_call_fn_1785ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5B3
&__inference_signature_wrapper_25052971input_1ћ
#__inference__wrapped_model_25052956m#0б-
&б#
!і
input_1         
ф "3ф0
.
output_1"і
output_1         г
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1922a/б,
%б"
 і
inputs         
ф ".б+
$і!
0                  
џ ё
,__inference_biholomorphic_layer_call_fn_2011T/б,
%б"
 і
inputs         
ф "!і                  а
A__inference_dense_1_layer_call_and_return_conditional_losses_1842[/б,
%б"
 і
inputs         F
ф "%б"
і
0         F
џ x
&__inference_dense_1_layer_call_fn_1799N/б,
%б"
 і
inputs         F
ф "і         Fа
A__inference_dense_2_layer_call_and_return_conditional_losses_1930[/б,
%б"
 і
inputs         F
ф "%б"
і
0         F
џ x
&__inference_dense_2_layer_call_fn_1764N/б,
%б"
 і
inputs         F
ф "і         Fа
A__inference_dense_3_layer_call_and_return_conditional_losses_1849[#/б,
%б"
 і
inputs         F
ф "%б"
і
0         
џ x
&__inference_dense_3_layer_call_fn_1785N#/б,
%б"
 і
inputs         F
ф "і         Д
?__inference_dense_layer_call_and_return_conditional_losses_1772d8б5
.б+
)і&
inputs                  
ф "%б"
і
0         F
џ 
$__inference_dense_layer_call_fn_1834W8б5
.б+
)і&
inputs                  
ф "і         FГ
J__inference_kahler_potential_layer_call_and_return_conditional_losses_1997_#0б-
&б#
!і
input_1         
ф "%б"
і
0         
џ Ё
/__inference_kahler_potential_layer_call_fn_2006R#0б-
&б#
!і
input_1         
ф "і         б
&__inference_signature_wrapper_25052971x#;б8
б 
1ф.
,
input_1!і
input_1         "3ф0
.
output_1"і
output_1         