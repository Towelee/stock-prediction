	��� ��8@��� ��8@!��� ��8@	1��2o��?1��2o��?!1��2o��?"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'���� ��8@��K7�A8@A�E�����?Y���S㥫?*	     �W@2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapJ+��?!�����I@)J+��?1�����I@:Preprocessing2F
Iterator::Model����Mb�?!y�5��@@)�~j�t��?16��P^C9@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t��?!6��P^C)@)�~j�t��?16��P^C)@:Preprocessing2S
Iterator::Model::ParallelMap����Mb�?!y�5�� @)����Mb�?1y�5�� @:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����Mb`?!y�5�� @)����Mb`?1y�5�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B98.2 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��K7�A8@��K7�A8@!��K7�A8@      ��!       "      ��!       *      ��!       2	�E�����?�E�����?!�E�����?:      ��!       B      ��!       J	���S㥫?���S㥫?!���S㥫?R      ��!       Z	���S㥫?���S㥫?!���S㥫?JCPU_ONLY