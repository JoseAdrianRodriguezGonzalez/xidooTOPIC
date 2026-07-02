from modeling.presentation import TopicFormatter
def test_formatter():
    topics = {
        0: (["playa", "mar"], "[KEYWORDS] playa, mar"),
        1: (["hotel", "limpio"], "[KEYWORDS] hotel, limpio")
    }

    formatter = TopicFormatter()
    kw_list, report = formatter.format(topics)

    assert len(kw_list) == 2
    assert "Tópico: 0" in report
