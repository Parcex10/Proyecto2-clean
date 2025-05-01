import tensorflow as tf
import wandb
import os, glob, random

# ─── Config ─────────────────────────────────────────────────────────
PROJECT_NAME = "pix2pix-parachute"
BATCH_SIZE   = 1
IMG_WIDTH    = 256
IMG_HEIGHT   = 256
BUFFER_SIZE  = 400
EPOCHS       = 200

MASK_DIR     = "masks"
IMAGE_DIR    = "images"

# Inicializa W&B
wandb.init(project=PROJECT_NAME,
           config={
             "batch_size": BATCH_SIZE,
             "img_width" : IMG_WIDTH,
             "img_height": IMG_HEIGHT,
             "epochs"    : EPOCHS
           })
config = wandb.config

# ─── Funciones de carga y preprocesado ───────────────────────────────
def load_pair(mask_path):
    # mask_path: "masks/1.png" → real_path: "images/1.jpg"
    mask  = tf.io.decode_png(tf.io.read_file(mask_path), channels=3)
    real_path = tf.strings.regex_replace(mask_path, MASK_DIR, IMAGE_DIR)
    real_path = tf.strings.regex_replace(real_path, ".png", ".jpg")
    real  = tf.io.decode_jpeg(tf.io.read_file(real_path), channels=3)
    mask  = tf.cast(mask, tf.float32)
    real  = tf.cast(real, tf.float32)
    return mask, real

def resize(inp, tar, h, w):
    inp = tf.image.resize(inp, [h,w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tar = tf.image.resize(tar, [h,w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return inp, tar

def random_crop(inp, tar):
    stacked = tf.stack([inp, tar], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped[0], cropped[1]

def normalize(inp, tar):
    return (inp / 127.5) - 1, (tar / 127.5) - 1

@tf.function
def random_jitter(inp, tar):
    inp, tar = resize(inp, tar, IMG_WIDTH+30, IMG_HEIGHT+30)
    inp, tar = random_crop(inp, tar)
    if tf.random.uniform(()) > 0.5:
        inp = tf.image.flip_left_right(inp)
        tar = tf.image.flip_left_right(tar)
    return normalize(inp, tar)

def load_image_train(mask_path):
    inp, tar = load_pair(mask_path)
    inp, tar = random_jitter(inp, tar)
    return inp, tar

def load_image_test(mask_path):
    inp, tar = load_pair(mask_path)
    inp, tar = resize(inp, tar, IMG_HEIGHT, IMG_WIDTH)
    return normalize(inp, tar)

# ─── Dataset ─────────────────────────────────────────────────────────
mask_files = glob.glob(os.path.join(MASK_DIR, "*.*"))
random.shuffle(mask_files)
split = int(0.9 * len(mask_files))
train_masks = mask_files[:split]
test_masks  = mask_files[split:]

train_ds = tf.data.Dataset.from_tensor_slices(train_masks)
train_ds = (train_ds
    .map(load_image_train,   num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))

test_ds = tf.data.Dataset.from_tensor_slices(test_masks)
test_ds = (test_ds
    .map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE))

# ─── Modelos: U-Net (Generator) & PatchGAN (Discriminator) ──────────
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    seq  = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(filters, size, strides=2,
                                   padding='same',
                                   kernel_initializer=init,
                                   use_bias=False))
    if apply_batchnorm: seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.LeakyReLU())
    return seq

def upsample(filters, size, apply_dropout=False):
    init = tf.random_normal_initializer(0.,0.02)
    seq  = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=init,
                                            use_bias=False))
    seq.add(tf.keras.layers.BatchNormalization())
    if apply_dropout: seq.add(tf.keras.layers.Dropout(0.5))
    seq.add(tf.keras.layers.ReLU())
    return seq

def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    downs = [
      downsample(64,4,False), downsample(128,4),
      downsample(256,4),      downsample(512,4),
      downsample(512,4),      downsample(512,4),
      downsample(512,4),      downsample(512,4),
    ]
    ups = [
      upsample(512,4,True),   upsample(512,4,True),
      upsample(512,4,True),   upsample(512,4),
      upsample(256,4),        upsample(128,4),
      upsample(64,4),
    ]
    init = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,4,
                                           strides=2,padding='same',
                                           kernel_initializer=init,
                                           activation='tanh')
    x = inputs
    skips = []
    for d in downs:
        x = d(x); skips.append(x)
    skips = reversed(skips[:-1])
    for u,skip in zip(ups, skips):
        x = u(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    init = tf.random_normal_initializer(0.,0.02)
    inp = tf.keras.layers.Input(shape=[256,256,3], name='inp')
    tar = tf.keras.layers.Input(shape=[256,256,3], name='tar')
    x = tf.keras.layers.concatenate([inp, tar])
    d1 = downsample(64,4,False)(x)
    d2 = downsample(128,4)(d1)
    d3 = downsample(256,4)(d2)
    z1 = tf.keras.layers.ZeroPadding2D()(d3)
    c1 = tf.keras.layers.Conv2D(512,4,1,padding='valid',
                                kernel_initializer=init,
                                use_bias=False)(z1)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    l1 = tf.keras.layers.LeakyReLU()(b1)
    z2 = tf.keras.layers.ZeroPadding2D()(l1)
    last = tf.keras.layers.Conv2D(1,4,1,kernel_initializer=init)(z2)
    return tf.keras.Model(inputs=[inp,tar], outputs=last)

generator     = Generator()
discriminator = Discriminator()

# ─── Pérdidas & Optim ─────────────────────────────────────────────────
LAMBDA   = 100
loss_fn  = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def generator_loss(disc_out, gen_out, target):
    gan = loss_fn(tf.ones_like(disc_out), disc_out)
    l1  = tf.reduce_mean(tf.abs(target - gen_out))
    return gan + LAMBDA*l1

def discriminator_loss(d_real, d_gen):
    real_loss = loss_fn(tf.ones_like(d_real), d_real)
    gen_loss  = loss_fn(tf.zeros_like(d_gen), d_gen)
    return real_loss + gen_loss

# ─── Checkpoints ───────────────────────────────────────────────────────
ckpt_dir = "checkpoints_parachute"
ckpt = tf.train.Checkpoint(generator_optimizer=gen_opt,
                           discriminator_optimizer=disc_opt,
                           generator=generator,
                           discriminator=discriminator)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

# ─── Paso de entrenamiento ─────────────────────────────────────────────
@tf.function
def train_step(inp, tar):
    with tf.GradientTape() as tg, tf.GradientTape() as td:
        gen_out = generator(inp, training=True)
        d_real  = discriminator([inp, tar], training=True)
        d_gen   = discriminator([inp, gen_out], training=True)
        g_loss  = generator_loss(d_gen, gen_out, tar)
        d_loss  = discriminator_loss(d_real, d_gen)
    grads_g = tg.gradient(g_loss, generator.trainable_variables)
    grads_d = td.gradient(d_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(grads_g, generator.trainable_variables))
    disc_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    return g_loss, d_loss

# ─── Loop de entrenamiento ─────────────────────────────────────────────
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for step, (inp, tar) in enumerate(train_ds):
        gl, dl = train_step(inp, tar)
