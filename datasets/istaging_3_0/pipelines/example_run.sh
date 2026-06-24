#! /bin/bash

mkdir -pv ../test/output

## Anonymize data
din='/cbica/home/erusg/GitHub/gurayerus/nichart_data_validation/datasets/istaging_3_0/data_private'
cin='istaging_3_0_harmonized.csv'
dout='../test/data_anon'
python ./s1_anonymize.py --in_dir $din --in_csv $cin --out_dir $dout


## Select sample
din="../test/data_anon"
sin="test-s1/test-s1_list.csv"
dout="../test/data_sel"
python ./s2_select_sample.py --in_dir $din --in_sample_csv $sin --out_dir $dout


## Run pipelines
din="../test/data_sel"
dout="../test/output"
python ./s3_run_pipelines.py --input_dir $din --output_dir $dout -v

## Run QC
din="../test/data_sel"
dout="../test/output"
task='qc_dataset_test1'
python ./s4_run_qc.py --in_dir $din --task $task --out_dir $dout 
