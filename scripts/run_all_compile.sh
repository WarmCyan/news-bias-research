#!/bin/bash

echo "----------- Supporting NN arch search ----------- "

compile_results.py \
-e run_0/bias_w2v_avg_arch1,run_0/bias_w2v_avg_arch2,run_0/bias_w2v_avg_arch3,run_0/bias_w2v_avg_arch4,run_0/bias_w2v_avg_arch5,run_0/bias_w2v_avg_arch6,run_0/bias_w2v_avg_arch7,run_0/bias_w2v_avg_arch8 \
-o bias_avg_arch \
--caption "Neural network achitecture search" \
--column-replacements "bias_w2v_avg_arch1=Arch 1,bias_w2v_avg_arch2=Arch 2,bias_w2v_avg_arch3=Arch 3,bias_w2v_avg_arch4=Arch 4,bias_w2v_avg_arch5=Arch 5,bias_w2v_avg_arch6=Arch 6,bias_w2v_avg_arch7=Arch 7,bias_w2v_avg_arch8=Arch 8" \
--final

echo "----------- Supporting LSTM arch search ----------- "

compile_results.py \
-e seq/bias_w2v_seq_arch1,seq/bias_w2v_seq_arch2,seq/bias_w2v_seq_arch3,seq/bias_w2v_seq_arch4 \
-o bias_seq_arch \
--caption "LSTM architecture search" \
--column-replacements "bias_w2v_seq_arch1=Arch 1,bias_w2v_seq_arch2=Arch 2,bias_w2v_seq_arch3=Arch 3,bias_w2v_seq_arch4=Arch 4" \
--final

echo "----------- Supporting Avg w stddev (rel) ----------- "

compile_results.py \
-e run_0/rel_svm_w2v_avg,run_0/rel_svm_w2v_avg_std,run_0/rel_w2v_avg,run_0/rel_w2v_avg_std,run_1/rel_svm_w2v_avg,run_1/rel_svm_w2v_avg_std,run_1/rel_w2v_avg,run_1/rel_w2v_avg_std \
-o rel_avg \
--caption "Average vector versus average and standard deviation vector."
--column-replacements "rel_svm_w2v_avg=Average,rel_svm_w2v_avg_std=Average + \\sigma" \
--final

echo "----------- Reliability Embedding ----------- "

compile_results.py \
-e run_0/rel_svm_w2v_avg,run_0/rel_svm_glove_avg,run_0/rel_svm_ft_avg,run_0/rel_svm_tfidf \
-o rel_embed \
--caption "Embedding comparisons on predicting reliability." \
--column-replacements "rel_svm_w2v_avg=Word2Vec,rel_svm_glove_avg=GloVe,rel_svm_ft_avg=FastText,rel_svm_tfidf=TF-IDF" \
--final

echo "----------- Reliability Selection Set ----------- "

compile_results.py \
-e run_0/rel_svm_w2v_avg,run_0/rel_svm_w2v_avg_ng,run_0/rel_svm_w2v_avg_mbfc \
-o rel_sel \
--caption "Reliability selection set comparison." \
--column-replacements "rel_svm_w2v_avg=Combined,rel_svm_w2v_avg_mbfc=MB/FC,rel_svm_w2v_avg_ng=NewsGuard" \
--final

echo "----------- Reliability Sentics ----------- "

compile_results.py \
-e run_0/rel_svm_w2v_avg,run_0/rel_svm_w2v_limit_avg,run_0/rel_svm_w2v_sentic_avg,run_0/rel_svm_w2v_sentic_full_avg,run_0/rel_svm_w2v_sentic_avg_std,run_0/rel_svm_w2v_sentic_full_avg_std \
-o rel_sentic \
--caption "Sentics incorporation." \
--column-replacements "rel_svm_w2v_avg=No sentics,rel_svm_w2v_limit_avg=Limited,rel_svm_w2v_sentic_avg=Sentics,rel_svm_w2v_sentic_full_avg=Padded sentics" \
--final

echo "----------- Reliability Algorithm ----------- "

compile_results.py \
-e seq/rel_w2v_seq,run_0/rel_svm_w2v_avg,run_0/rel_w2v_avg \
-o rel_alg \
--caption "ML algorithm comparison." \
--column-replacements "rel_svm_w2v_avg=SVM,rel_w2v_avg=NN,rel_w2v_seq=LSTM" \
--final

echo "----------- Bias Embedding ----------- "

compile_results.py \
-e run_0/bias_svm_tfidf,run_0/bias_svm_w2v_avg,run_0/bias_svm_glove_avg,run_0/bias_svm_ft_avg \
-o bias_embed \
--caption "Embedding comparison in biased versus unbiased predictions." \
--column-replacements "bias_svm_w2v_avg=Word2Vec,bias_svm_glove_avg=GloVe,bias_svm_ft_avg=FastText,bias_svm_tfidf=TF-IDF" \
--final

echo "----------- Bias Selection Set ----------- "

compile_results.py \
-e run_0/bias_svm_w2v_avg,run_0/bias_svm_w2v_avg_allsides_all,run_0/bias_svm_w2v_avg_mbm,run_0/bias_svm_w2v_avg_below20flip,run_0/bias_svm_w2v_avg_below25flip \
-o bias_sel \
--caption "Bias labeling selection set comparison." \
--column-replacements "bias_svm_w2v_avg=Combined,bias_svm_w2v_avg_allsides_all=AllSides,bias_svm_w2v_avg_mbm=MBM,bias_svm_w2v_avg_below20flip=Combined (\\textless20\\%),bias_svm_w2v_avg_below25flip=Combined (\\textless25\\%)" \
--final

echo "----------- Bias Sentics ----------- "

compile_results.py \
-e run_0/bias_svm_w2v_avg,run_0/bias_svm_w2v_limit_avg,run_0/bias_svm_w2v_sentic_avg,run_0/bias_svm_w2v_sentic_full_avg,run_0/bias_svm_w2v_sentic_avg_std,run_0/bias_svm_w2v_sentic_full_avg_std \
-o bias_sentic \
--caption "Sentics incorporation." \
--column-replacements "bias_svm_w2v_avg=No sentics,bias_svm_w2v_limit_avg=Limited,bias_svm_w2v_sentic_avg=Sentics,bias_svm_w2v_sentic_full_avg=Padded sentics" \
--final


echo "----------- Bias Algorithm ----------- "

compile_results.py \
-e run_0/bias_svm_w2v_avg,seq/bias_w2v_seq,run_0/bias_w2v_avg \
-o bias_alg \
--caption "ML algorithm comparison." \
--column-replacements "bias_svm_w2v_avg=SVM,bias_w2v_avg=NN,bias_w2v_seq=LSTM" \
--final
