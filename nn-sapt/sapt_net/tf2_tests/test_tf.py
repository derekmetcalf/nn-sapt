import tensorflow as tf

atoms = tf.constant(["O", "H", "C", "N"], name="atoms")
atomtype = tf.constant("H", name="atomtype")
a = tf.equal(atomtype,atoms)
with tf.Session() as sess:
    print(a.eval())
    writer = tf.summary.FileWriter("summaries", sess.graph)
