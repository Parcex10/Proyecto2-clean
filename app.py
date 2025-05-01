import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Reconstruye la U-Net
OUTPUT_CHANNELS = 3
def downsample(filters,size,apply_batchnorm=True):
    init = tf.random_normal_initializer(0.,0.02)
    seq  = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(filters,size,2,'same',init,use_bias=False))
    if apply_batchnorm: seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.LeakyReLU())
    return seq
def upsample(filters,size,apply_dropout=False):
    init = tf.random_normal_initializer(0.,0.02)
    seq  = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2DTranspose(filters,size,2,'same',init,use_bias=False))
    seq.add(tf.keras.layers.BatchNormalization())
    if apply_dropout: seq.add(tf.keras.layers.Dropout(0.5))
    seq.add(tf.keras.layers.ReLU())
    return seq
def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    downs = [downsample(64,4,False), downsample(128,4),
             downsample(256,4), downsample(512,4),
             downsample(512,4), downsample(512,4),
             downsample(512,4), downsample(512,4)]
    ups   = [upsample(512,4,True), upsample(512,4,True),
             upsample(512,4,True), upsample(512,4),
             upsample(256,4), upsample(128,4),
             upsample(64,4)]
    init  = tf.random_normal_initializer(0.,0.02)
    last  = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,4,2,'same',init,activation='tanh')
    x = inputs; skips=[]
    for d in downs: x= d(x); skips.append(x)
    skips = reversed(skips[:-1])
    for u,s in zip(ups,skips): x = u(x); x=tf.keras.layers.Concatenate()([x,s])
    x= last(x)
    return tf.keras.Model(inputs=inputs,outputs=x)

# Carga pesos
generator = Generator()
ckpt = tf.train.Checkpoint(generator=generator)
latest = tf.train.latest_checkpoint('checkpoints_parachute')
if latest: ckpt.restore(latest).expect_partial()
else: raise FileNotFoundError("No encontré checkpoints_parachute/")

# Pre/post procesado
def preprocess(img):
    img = img.resize((256,256))
    arr = np.array(img).astype(np.float32)
    return (arr/127.5)-1

def postprocess(t):
    img = (t*0.5+0.5)*255.
    return np.clip(img,0,255).astype(np.uint8)

def predict(mask_np):
    pil = Image.fromarray(mask_np.astype(np.uint8)).convert('RGB')
    x = preprocess(pil)[None,...]
    y = generator(x,training=False)[0]
    return postprocess(y)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='numpy',label="Boceto 256×256"),
    outputs=gr.Image(type='numpy',label="Predicción Realista"),
    title="Pix2Pix Paracaídas",
    description="Transforma un boceto en máscara a una foto realista",
    allow_flagging=False
)

if __name__=="__main__":
    iface.launch(server_name='0.0.0.0',server_port=7860)
