from langchain_text_splitters import CharacterTextSplitter

document = """
Once upon a time, in a land far, far away, there lived a brave knight named Sir Cedric. Sir Cedric was known throughout the kingdom for his valor and kindness. He embarked on many adventurous quests, slaying dragons, fighting bandits, and rescuing those in distress. The villagers often sang songs of his bravery and celebrated his victories with grand feasts.
One day, Sir Cedric heard about a ferocious dragon terrorizing a distant village. This dragon, known as Valthor, had been causing havoc, burning crops, and stealing livestock. Determined to save the villagers, Sir Cedric set off on his trusty steed, Thunder, to confront Valthor.
The journey was long and arduous. Sir Cedric traversed dense forests, crossed raging rivers, and climbed steep mountains. Along the way, he encountered various challenges. He helped a group of merchants fend off a group of thieves and aided an old woman in finding her lost cat. Each act of kindness bolstered his resolve and strengthened his spirit.

As Sir Cedric continued his journey, he encountered a wise old hermit living in a secluded cave. The hermit, intrigued by Sir Cedric's quest, offered him valuable advice and a magical amulet for protection. Grateful for the hermit's kindness, Sir Cedric continued on his path with renewed determination.

After days of travel, Sir Cedric finally arrived at the village tormented by Valthor. The sight was heartbreaking; homes were reduced to ashes, and the villagers lived in constant fear. Sir Cedric vowed to end their suffering. He devised a plan to lure the dragon out of its lair and into an open field where he could engage it in battle.

As the sun set, casting an orange glow over the village, Sir Cedric stood ready. The ground trembled as Valthor emerged, its scales glistening menacingly. With a roar, the battle began. Sir Cedric fought valiantly, his sword clashing against the dragon's fiery breath. After a fierce struggle, he managed to strike a critical blow, vanquishing Valthor once and for all.

The villagers rejoiced, their hero had saved them. Sir Cedric's tale of bravery spread far and wide, and he returned home, not only as a knight but as a legend.

Upon his return to the kingdom, Sir Cedric was celebrated with a grand parade. The king and queen bestowed upon him the highest honors and declared a day of festivities in his name. Sir Cedric used this opportunity to share the wisdom he had gained from his journeys, encouraging others to show courage, kindness, and resilience in the face of adversity.

Inspired by Sir Cedric's example, many young knights and adventurers set out on their own quests, determined to make a difference in the world. Sir Cedric continued to serve the kingdom, not only as a protector but also as a mentor and guide to the next generation of heroes.

Years passed, and Sir Cedric's legend grew even more. Tales of his bravery and selflessness were told in every corner of the land. He had faced countless challenges, but his unwavering dedication to justice and compassion had made him a true hero. As he grew older, Sir Cedric retired to a peaceful countryside estate, where he spent his days reflecting on his adventures and the lives he had touched.

Even in retirement, Sir Cedric remained a beloved figure in the kingdom. He would often be visited by those seeking his wisdom and guidance. He continued to inspire and uplift those around him, proving that true heroism is not just about grand deeds but also about the positive impact one can have on others' lives.

And so, Sir Cedric's legacy endured, reminding everyone that courage, kindness, and integrity are the qualities that define a true hero.
"""

# Create a CharacterTextSplitter instance with specified encoding
print('\n\n################################################')
print(f'\nUsing the Splitting Separator: \\n\\n')
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

# Split the document into chunks
texts = text_splitter.split_text(document)

# Print the resulting chunks
for i, chunk in enumerate(texts):
    print(f"\nChunk {i+1}:\n{chunk}")


# Create a CharacterTextSplitter instance with specified encoding
print('\n\n################################################')
print(f'\nUsing the Splitting Separator: \\n')
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

# Split the document into chunks
texts = text_splitter.split_text(document)

# Print the resulting chunks
for i, chunk in enumerate(texts):
    print(f"\nChunk {i+1}:\n{chunk}")

# Create a CharacterTextSplitter instance with specified encoding
print('\n\n################################################')
print(f'\nUsing the Splitting Separator: .')
text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=1000,
    chunk_overlap=200
)

# Split the document into chunks
texts = text_splitter.split_text(document)

# Print the resulting chunks
for i, chunk in enumerate(texts):
    print(f"\nChunk {i+1}:\n{chunk}")