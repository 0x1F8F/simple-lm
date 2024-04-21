from simple_bigram import tokenizer


def test_tokenizer():
    model = tokenizer.Tokenizer("data/dataset.csv")
    text = "hello world"
    a = model.encoder(text)
    b = model.decoder(a)
    assert text==b