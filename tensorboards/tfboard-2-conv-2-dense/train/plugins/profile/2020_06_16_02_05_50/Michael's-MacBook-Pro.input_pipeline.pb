	y�&1L>@y�&1L>@!y�&1L>@	�a�Y���?�a�Y���?!�a�Y���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$y�&1L>@+�����?A/�$�>@YH�z�G�?*	      k@2F
Iterator::Model��MbX�?!������F@)�I+��?1�%���^D@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate+�����?!�8��8B@)㥛� ��?1_B{	��@@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�Q���?!B{	�%40@)�� �rh�?1���^B{/@:Preprocessing2S
Iterator::Model::ParallelMap�I+��?!�%���^@)�I+��?1�%���^@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{�G�zt?!Lh/���@){�G�zt?1Lh/���@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapˡE����?!���^B�B@)����Mbp?1�Kh/��?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����MbP?!�Kh/��?)����MbP?1�Kh/��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	+�����?+�����?!+�����?      ��!       "      ��!       *      ��!       2	/�$�>@/�$�>@!/�$�>@:      ��!       B      ��!       J	H�z�G�?H�z�G�?!H�z�G�?R      ��!       Z	H�z�G�?H�z�G�?!H�z�G�?JCPU_ONLY