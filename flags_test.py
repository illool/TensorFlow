import tensorflow as tf

#tensorflow的变量定义有4种类型，整型，字符，浮点，布尔。
#第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_string('str_name', 'def_v_1',"字符")
tf.app.flags.DEFINE_integer('int_name', 10,"整型")
tf.app.flags.DEFINE_float('float_name', 10.0,"浮点")
tf.app.flags.DEFINE_boolean('bool_name', False, "布尔")

FLAGS = tf.app.flags.FLAGS

#必须带参数，否则：'TypeError: main() takes no arguments (1 given)';   main的参数名随意定义，无要求
def main(_):
    print(FLAGS.str_name)
    print(FLAGS.int_name)
    print(FLAGS.float_name)
    print(FLAGS.bool_name)

#python flags_test.py --str_name test_str --int_name 99 --float_name 99.0 --bool_name True
if __name__ == '__main__':
    tf.app.run()  #执行main函数
