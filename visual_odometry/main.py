class VisualOdometryPipeline:

    def run(self):
        print("Driver Method for Visual Odometry Pipeline")
        raise NotImplementedError()

def main():
    pipeline = VisualOdometryPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
