?	BA)Z??@BA)Z??@!BA)Z??@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-BA)Z??@9'??>?#@1f?O70˒@A???R%??IƆn?Z#@*	}?5^??i@2F
Iterator::Model??2nj???!?"1???D@)m;m?Ʃ?1?P??q8@:Preprocessing2U
Iterator::Model::ParallelMapV2???N?z??!?D??0@)???N?z??1?D??0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	S?K???!?"d??,@)	S?K???1?"d??,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat䞮?Xl??!HM?? l2@)?d ??Ɲ?1?????=,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip,e?X??!??u}M@)?#??S ??1?96<Ŏ*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??.¬?!?O?¡F;@)?}?u?r??18???k?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;:?Fv???!?h?Ƈ4@)?B??f??1>\\?	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor>??j?#??!?Q?84@)>??j?#??1?Q?84@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ??E1???Q?:??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	9'??>?#@9'??>?#@!9'??>?#@      ??!       "	f?O70˒@f?O70˒@!f?O70˒@*      ??!       2	???R%?????R%??!???R%??:	Ɔn?Z#@Ɔn?Z#@!Ɔn?Z#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ??E1???y?:??X@?"g
;gradient_tape/model/conv3d_26/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2k̘?????!k̘?????"g
;gradient_tape/model/conv3d_27/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2e7q?m??!????1??"E
)gradient_tape/model/up_sampling3d_3/splitSplitV??z?????!<fP???"f
:gradient_tape/model/conv3d_2/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2=Л????!&??H????"f
:gradient_tape/model/conv3d_1/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2A>?Y??!Hu??????"4
model/conv3d_27/Conv3DConv3DA?X;???!r?X???0"4
model/conv3d_26/Conv3DConv3Dr?????!????X??0"g
;gradient_tape/model/conv3d_22/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2????????!?*??d??"e
:gradient_tape/model/conv3d_26/Conv3D/Conv3DBackpropInputV2Conv3DBackpropInputV2?D]?????!??#S??"g
;gradient_tape/model/conv3d_23/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2?=?݅???!???<7(??Q      Y@Y?ĉth??a?v?/?X@q?^?^_??yP?C??1?"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 