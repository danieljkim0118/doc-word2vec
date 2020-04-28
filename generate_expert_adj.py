topics = ['Accounting', 'Advertising', 'Arts', 'Banking', 'BusinessServices', 'Chemicals', 'Communications-Media', 'Consulting', 'Education', 'Engineering', 'Fashion', 'Government', 'Internet', 'Law', 'Marketing', 'Non-Profit', 'Publishing', 'Religion', 'Science', 'Student', 'Technology']

name = input("Enter your name -- ")

print("There are a total of " + str(len(topics) ** 2) + " topic pairs")
print("Are the following topics related (y/n):")

sim_dict = {}
finished_topics = []

for topic1 in topics:
    for topic2 in [x for x in topics if x != topic1]:
        if topic2 not in finished_topics:
            valid_input = False
            while not valid_input:
                similarity = input(topic1 + " and " + topic2 + " -- ")

                try:
                    if similarity == "y" or similarity == "n":
                        valid_input = True
                except:
                    valid_input = False

            try:
                sim_dict[topic1][topic2] = similarity
            except:
                sim_dict[topic1] = {}
                sim_dict[topic1][topic2] = similarity
    finished_topics.append(topic1)


input("Once you press enter, copy the following and paste it into chat: ")
print()
print(name)
print(sim_dict)