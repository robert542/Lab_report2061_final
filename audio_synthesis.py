import streamlit as st
import numpy as np
import scipy.io.wavfile
import sounddevice as sd
import threading

### ALL ANALYSIS DONE ON TRIMMED AUDIO FILES #############
### TRIAL 1: keep\trimmed\guitar_tr1_trimmed_ns.wav  ####
### TRIAL 2: keep\trimmed\guitar_tr2_trimmed_ns.wav  ####
### TRIAL 3: keep\trimmed\guitar_tr3_trimmed_ns.wav  ####
#########################################################

def play_sound(sound, sample_rate):
    sd.play(sound, sample_rate)
    sd.wait()

def stop_sound():
    sd.stop()

def synthesize_sound(base_freq, sample_rate, max_harmonic, volume, add_fall_off, b_value:float|None=None):
    duration_s = 8
    sample_number = np.arange(duration_s * sample_rate)
    total = np.zeros(duration_s * sample_rate)
    #change nth frequency function
    if b_value is None:
        freq = lambda n: base_freq*n
    else:
        freq = lambda n: base_freq * n * np.sqrt(1 + b_value * (n ** 2))
        
    for i in range(1, 2 * max_harmonic, 2):
        if i != 1:
            amp = (0.5 / ((i**2))) * volume  
        else:
            amp = 0.5 * volume  
        phase = 0 
        total += amp * np.sin(2 * np.pi * sample_number * freq(i) / sample_rate + phase)
    
    if add_fall_off:
        decay_shape = np.exp(-sample_number / (1.0 * sample_rate))
        attack_shape = 1 - np.exp(-sample_number / (0.01 * sample_rate))
        total *= decay_shape * attack_shape

    return total

def main():
    st.title("Waveform Synthesis with Inharmonicity")
    
    uploaded_file = st.file_uploader("Upload a file", type=["wav"])
    sample_rate = 44100  
    if uploaded_file is not None:
        sample_rate, data = scipy.io.wavfile.read(uploaded_file)
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
        st.write(f"Sample rate: {sample_rate}")
        volume = st.slider("Volume for original sound", 0.1, 2.0, 1.0, 0.1)
        if st.button('Play Original Sound'):
            threading.Thread(target=play_sound, args=(data * volume, sample_rate)).start()

    base_freq = st.slider("Base Frequency", 80.0, 130.0, 100.0,0.01)
    max_harmonic = st.slider("Max Harmonic (n)", 1, 160, 3)
    volume = st.slider("Volume for synthesized sound", 0.1, 2.0, 1.0, 0.1)
    add_fall_off = st.checkbox("Add Fall Off")
    
    if uploaded_file is not None:
        synthesized_sound = synthesize_sound(base_freq, sample_rate, max_harmonic, volume, add_fall_off)
        synthesized_sound = synthesized_sound.astype(np.float32)
        if st.button('Play Synthesized Sound'):
            threading.Thread(target=play_sound, args=(synthesized_sound, sample_rate)).start()

    st.button("Stop Sound", on_click=stop_sound)

if __name__ == "__main__":
    main()
