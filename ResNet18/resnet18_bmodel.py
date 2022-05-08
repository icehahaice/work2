import bmnetp

def main():
    bmnetp.compile(
        model="./resnet18_traced.zip",
        shapes=[[1, 3, 224, 224]],
        net_name="resnet18",
        outdir="./ResNet18_bmodel",
        target="BM1684",
        opt=2
    )

if __name__ == '__main__':
    main()

