import gzip
import csv

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

#read cluster result from the csv file
def read_cluster():
    ids = []
    all_clusters = []
    with open('cluster_result.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            clusters = []
            for i, item in enumerate(row):
                #print (item)
                if i == 0: ids.append(item)
                else: clusters.append(item)

            #print (scores)
            all_clusters.append(clusters)
    print (len(ids))
    print (len(all_clusters))
    read_review(ids, all_clusters)
    #
    # with open(filename, 'w') as f:
    #     f.write("\n")

#read reviews, then compare to the cluster results in order to output
def read_review(ids, all_clusters):
    with open('review_cluster.txt', 'w') as f:
        f.write("")


    id_index = -1
    all_review = ""
    reviewer_id = {}
    prev_product = ''
    product_id = ''
    product_num = 0  # the number of products, starts from 1
    num_review = 0  # for counting number of reviews for each product
    review_list = []
    review_person = []
    review_each_movie = []
    possible_words = []
    for i, review in enumerate(parse("reviews_Movies_and_TV_5.json.gz")):
        product_id = review['asin']
        #Choose a specific product number, first 0005019281, second product 0005119367, 
        this_review = review['reviewText']
        aspects =  ["acting", "directing", "scene", "character", "story"]
        if (product_id != prev_product):
            if (prev_product != ''):
                for i, id in enumerate(ids):
                    if (id == prev_product):
                        #print (id + " " +prev_product)
                        id_index = i
                        break
                print(len(review_person))


                cluster_num = 0
                isContinue = True;

                while isContinue:
                    cluster_words = [[], [], [], [], []]
                    aspect_socres = [0, 0, 0, 0, 0]
                    isContinue = False
                    toWrite = ''
                    isStart = True
                    num = [0, 0, 0, 0, 0]
                    for j, cluster in enumerate(all_clusters[id_index]):
                        #print (str(cluster))
                        if cluster == str(cluster_num):
                            #print ("approve" + str(j))
                            isContinue = True
                            if isStart :
                                isStart = False
                            toWrite += review_person[j] + "\n"
                            #print (str(j))
                            file = open('score.csv', 'r')
                            reader = csv.reader(file)
                            for row in reader:
                                #print (row[0] +" " + prev_product + " " )
                                # print(row[1] + " " + str(j) + " ")
                                # print ( row[1] == str(j))
                                if row[0] == prev_product and row[1] == str(j):
                                    #print (row[0] + " " +str(j))

                                    for m in range(2, 7):
                                        toWrite += aspects[m - 2] + ": " + row[m] + "  "
                                        #toWrite += (row[7+m-2]) + "\n"
                                        if "None" not in row[m]:
                                            num[m-2] += 1
                                            aspect_socres[m-2] += float(row[m])
                                        cluster_words[m-2].append(row[7+m-2])
                                    toWrite += "\n"
                                    break



                            file.close()
                            # for row in reader:
                            #     if row[1] == str(id_index) and row[0] == prev_product:
                            #         for k in range(2, len(row)):
                            #             print (k-2)
                            #             toWrite += aspects[k-2] + ": "
                            #             toWrite += row[k] + "  "
                            #         toWrite += "\n"
                            #         break
                            toWrite += "\n"
                    if isContinue:
                        toWrite += "\n"
                        #print (toWrite)
                        with open('review_cluster.txt', 'a') as f:
                            f.write("Cluster: " + str(cluster_num) + "\n")
                            for z, aspect in enumerate(aspects):
                                if num[z] != 0: score = str(aspect_socres[z]/num[z])
                                else: score =None
                                f.write(aspect + ": " + str(score) + "  ")
                                #should actually be counted number
                                words = " ".join(cluster_words[z])
                                print (words)
                                f.write("  ".join(set(words.split())))
                                f.write("\n")
                            f.write("\n")
                            f.write(toWrite)
                    cluster_num += 1;


            with open('review_cluster.txt', 'a') as f:
                f.write("*" * 30 + str(product_num) + "th product" + product_id + "*" * 30 + "\n")
            print("*" * 30, product_num, "th product", product_id, "*" * 30)
            num_review = 0
            # with open(filename, 'a') as f:
            #     f.write("*" * 30+str(product_num) + "th product" +product_id +"*" * 30)
            #     f.write("\n")

        if (product_id == prev_product): num_review += 1

        if prev_product == '': #initialte the first one

            product_num += 1
            prev_product = product_id
            all_review += review['reviewText']
            all_review += " \n"
            review_person.append(review['reviewText'])
            #print prev_product, product_id
        elif product_id == prev_product:    #same product? then add reviews together
            all_review += review['reviewText']
            all_review += " \n"
            review_person.append(review['reviewText'])
            #print prev_product, product_id
        elif product_id != prev_product: #product_id no equals to previous product
            review_person.append(review['reviewText'])



            ######This should always happen in the end######
            review_list.append(all_review) #each element is all the reviews under a movie
            all_review = review['reviewText']  # to start new one
            prev_product = product_id
            product_num += 1    #product number plus one"
            #review_each_movie.append(review_person)
            review_person = []
            review_person.append(review['reviewText'])
    
        if product_num == len(ids):    #after the product_num, break the for loop, always -1
            break


read_cluster()





