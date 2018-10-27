import argparse
import config

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    #训练参数
    parser.add_argument('--batch_size',    type=int,   default=config.batch_size)
    parser.add_argument('--keep_prob',     type=float, default=config.keep_prob)
    parser.add_argument('--logfrequency',  type=int,   default=config.logfrequency)
    parser.add_argument('--Max_step',      type=int,   default=config.Max_step)
    parser.add_argument('--Max_epoch',     type=int, default=config.Max_epoch)
    parser.add_argument('--embed_dim',     type=int,   default=config.embed_dim)
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate)
    parser.add_argument('--oridata_dim',   type=int,   default=config.oridata_dim)
    parser.add_argument('--valid_switch',  type=int, default=config.valid_switch)
    parser.add_argument('--model_flag',    type=str, default=config.model_flag)
    # 路径和文件配置
    parser.add_argument('--encod_train_path',      type=str, default=config.encod_train_path)
    parser.add_argument('--encod_vaild_path',      type=str, default=config.encod_vaild_path)
    parser.add_argument('--encod_test_path',       type=str, default=config.encod_test_path)
    parser.add_argument('--dictsizefile',          type=str, default=config.dictsizefile)
    parser.add_argument('--model_ouput_dir',       type=str, default=config.model_ouput_dir)
    parser.add_argument('--summary_dir',           type=str, default=config.summary_dir)
    parser.add_argument('--dnn_log_file',          type=str, default=config.dnn_log_file)
    parser.add_argument('--dnn_log_dir',           type=str, default=config.dnn_log_dir)
    parser.add_argument('--dnn_log_path',          type=str, default=config.dnn_log_path)
    parser.add_argument('--encod_cat_index_begin', type=int, default=config.encod_cat_index_begin)
    parser.add_argument('--encod_cat_index_end',   type=int, default=config.encod_cat_index_end)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print (dir(FLAGS))
    for i in dir(FLAGS):
        print ('{}:{}'.format(i,getattr(FLAGS,i)))