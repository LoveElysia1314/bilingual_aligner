#!/usr/bin/env python3
"""
Test script for bilingual text alignment repair.
Hardcodes two text segments and uses the API to perform repair and save results.
"""

import tempfile
import os
from bilingual_aligner.api import TextAligner


def main():
    # Hardcoded source and target texts
    source_text = """总觉得听到副声道了。她以俄语在「久世同学」旁边加注小字。 
被她这么说的政近也想大声辩解，但是既然假装听不懂俄语就无法多说什么。 
总之要是反驳说「可惜我是奶子星人」，艾莉莎心中的政近股价肯定创新低，全班女生也会抢著拋售政近股，所以到头来保持沉默应该才是正确答案。 
（不过啊～～仔细想想，我做的事情没那么严重吧？） 
艾莉莎的冷淡反应，使得政近内心冒出这种想法。 
政近摸艾莉莎的腿，原本就是艾莉莎自己的指示，因为害臊而往上踢的也是艾莉莎。 
看见内裤是不可抗力的结果，后来以死前讯息的方式提醒应该是多此一举，不过这也是贴心暗示自己不在意艾莉莎动粗抵抗……政近不太能接受只有自己被视为错的一方。 
不过他知道，男生在这种状况往往屈居劣势，所以决定别乱说话直接道歉。 
「那个，对不起。刚才各方面惹你生气。」 
「……我不介意啊？毕竟我也有错，而且我已经没生气了啊？」 
政近「那你为什么看起来这么不高兴啊～～」的内心疑问，以及全班同学偷听之后「绝对在说谎……」的内心感想重叠在一起。 """

    target_text = """It felt as if he'd heard a subchannel. She'd added small Russian words next to "Kuse-kun."
Hearing this, Masachika wanted to shout his defense,
but since he'd pretended not to understand Russian, he couldn't say much. After all, if he retorted, "Unfortunately, I'm a boob person," Masachika's stock in Alya's eyes would plummet to an all-time low, and every girl in the class would rush to dump their Masachika shares. In the end, silence was probably the right answer.
"But... when you think about it carefully, what I did wasn't really that serious, was it?"
Alya's cold response triggered this thought within Masachika.
After all, touching Alya's leg had originally been her own instruction, and it was she herself who kicked upward out of shyness. Seeing her underwear was an irresistible force, and reminding her afterward via a "last message" was probably unnecessary—but it was also a considerate hint that she didn't mind his rough resistance... Masachika just couldn't accept being the only one seen as at fault.
However, he knew that boys were often at a disadvantage in such situations, so he decided not to say anything rash and just apologize directly.
"Um... I'm sorry. I upset you in various ways just now."
"...I don't really mind, do I? After all, I was also at fault, and I've already gotten over being angry, haven't I?"
Masachika's inner question, "Then why do you look so unhappy~," overlapped with the classmates' inner thoughts after eavesdropping, "She's definitely lying..."
"""

    # Create temporary files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as src_file:
        src_file.write(source_text)
        src_path = src_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tgt_file:
        tgt_file.write(target_text)
        tgt_path = tgt_file.name

    try:
        # Initialize aligner and perform repair
        aligner = TextAligner(src_path, tgt_path)
        result = aligner.repair()

        # Save results using the API
        output_dir = tempfile.mkdtemp()
        aligner.save_results(result, output_dir)

        # Print a simple report
        aligner.print_report(result, output_dir)

    finally:
        # Clean up temporary files
        try:
            os.unlink(src_path)
            os.unlink(tgt_path)
        except:
            pass


if __name__ == "__main__":
    main()
