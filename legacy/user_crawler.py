##Tweepy
import tweepy as tw

##Crawler de usuários:
def user_crawler(df):
    ###Twitter Developer Keys:
    # Twitter Developer Keys for crawler_pivic app
    consumer_key = '<key>'
    consumer_secret = '<key>'
    access_token = '<key>'
    access_token_secret = '<key>'

    ###Autgenticação entre Twitter Developer e este script:
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    ###Ínicio da recuperação de usuários:
    sleepTime = 2
    id_list = []
    user_screen_name = []
    user_id = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        id_list.append(row.tweet_id)
    for tweet_id in tqdm(id_list, total=len(id_list)):
        try:
            tweetFetched = api.get_status(tweet_id)
            #Para saber o que poder ser recuperado pelo objeto status ou objeto user, pesquisar.
            #print("User screen name: " + tweetFetched.user.screen_name)
            #print("User id: " + tweetFetched.user.id_str)
            #print("User name: " + tweetFetched.user.name)        
            user_screen_name.append(tweetFetched.user.screen_name)
            user_id.append(tweetFetched.user.id_str)
            #user_name.append(tweetFetched.user.name)
            #trainingDataSet.append(tweetFetched)
            time.sleep(sleepTime)        
        except:
            user_screen_name.append('invalid_user')
            user_id.append('invalid_user')
            #user_name.append('invalid_user')
            #print("Inside the exception - no:2")
            continue
        
    try:
        df.insert(3,'user_screen_name', user_screen_name)
    except:
        df['user_screen_name'] = user_screen_name
    try:
        df.insert(4,'user_id', user_id)
    except:
        df['user_id'] = user_id