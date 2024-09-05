from model import setup, train, run

if __name__ == '__main__':
    setup('data/licht')
    train()
    run('data/test/other-0007.mp3')