from vdc_vixtss_module import generate_voice


if __name__ == "__main__":
    out_wav = generate_voice("hello")
    print("*****out_wav ", out_wav)
