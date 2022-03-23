from hand import Hand


if __name__ == "__main__":
    try:
        detector = Hand()
        detector.main()
    except Exception as e:
        print(e)