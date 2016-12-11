THEANO_FLAGS="floatX=float32,device=gpu0,cuda.root=/usr/local/cuda,on_unused_input=ignore,optimizer=fast_compile"  python main.py --embedding data/yelp13New/vectors   --train data/yelp13New/train.txt    --dev data/yelp13New/dev.txt  --test data/yelp13New/test.txt --save model/yelp13New  --hidden_dim 50 --act relu  --pooling 1 --eval_period 500
# --learning sgd
