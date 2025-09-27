if __name__ == '__main__':
    # gen_model = GenModel()
    # # print("GenModel construction:ok")
    # gen_model.save_h_params('h_params.pkl')
    # # print("save_h_params:ok")
    # gen_model.load_h_params('h_params.pkl')
    # gen_model.load_h_params('h0_params.pkl')
    # gen_model.load_h_params('hn_params.pkl')

    learn_model = LearnModel()
    # print("GenModel construction:ok")
    # learn_model.save_h_params('h_params.pkl')
    # print("save_h_params:ok")
    learn_model.load_h0_params('p_params.pkl')
    # learn_model.load_hn_params('h0_params.pkl')
    # learn_model.load_hn_params('hn_params.pkl')
    # print("load_h_params:ok")
    # gen_model.save_params('params.pkl')
    # print("save_params:ok")
    # gen_model.load_params('params.pkl')
    # print("load_params:ok")
    # try:
    #     gen_model.load_h_params('params.pkl')
    # except:
    #     print("Not load_h_params:ok")
    # try:
    #     gen_model.load_params('h_params.pkl')
    # except:
    #     print("Not load_params:ok")
    # gen_model.gen_params()
    # print("gen_params:ok")
    # print(gen_model.get_params())
    # x = gen_model.gen_sample(20)
    # print(x)
    # print("gen_sample:ok")
    # gen_model.save_sample("sample",20)
    # print("save_sample:ok")
    # gen_model.visualize_model()
    # gen_model.load_params('params.pkl')
    # print(gen_model.get_params())

    # learn_model = LearnModel()
    # print("LearnModel construction:ok")
    # learn_model.save_h0_params("h0_params.pkl")
    # print("save_h0_params:ok")
    # learn_model.load_h0_params("h0_params.pkl")
    # print("load_h0_params:ok")
    # learn_model.save_hn_params("hn_params.pkl")
    # print("save_hn_params:ok")
    # learn_model.load_hn_params("hn_params.pkl")
    # print("load_hn_params:ok")
    # learn_model.load_h0_params("hn_params.pkl")
    # print("load_h0_params from hn:ok")
    # learn_model.save_p_params("p_params.pkl")
    # print("save_p_params:ok")
    # learn_model.load_p_params("p_params.pkl")
    # print("load_p_params:ok")
    # try:
    #     learn_model.load_h0_params("p_params.pkl")
    # except:
    #     print("Not load_h0_params:ok")
    # try:
    #     learn_model.load_hn_params("p_params.pkl")
    # except:
    #     print("Not load_hn_params:ok")
    # try:
    #     learn_model.load_p_params("hn_params.pkl")
    # except:
    #     print("Not load_p_params:ok")

    # print(learn_model.get_h0_params())
    # print(learn_model.get_hn_params())
    # print(learn_model.get_p_params())

    # learn_model.update_posterior(x)
    # print("update_posterior:ok")
    # learn_model.visualize_posterior()
    # print("visualize_posterior:ok")
    # print(learn_model.get_h0_params())
    # print(learn_model.get_hn_params())
    # print(learn_model.get_p_params())

    # learn_model.calc_pred_dist(np.zeros(1))
    # print("calc_pred_dist: ok")
    # print(learn_model.get_h0_params())
    # print(learn_model.get_hn_params())
    # print(learn_model.get_p_params())

    # learn_model.reset_hn_params()
    # print("reset_hn_params: ok")
    # print(learn_model.get_h0_params())
    # print(learn_model.get_hn_params())
    # print(learn_model.get_p_params())

    # learn_model.update_posterior(x)
    # print(learn_model.get_h0_params())
    # print(learn_model.get_hn_params())
    # print(learn_model.get_p_params())
    # learn_model.overwrite_h0_params()
    # print("overwrite_h0_params: ok")
    # print(learn_model.get_h0_params())
    # print(learn_model.get_hn_params())
    # print(learn_model.get_p_params())



    # # learn_model = LearnModel(h0_mu_vec=np.array([1,1]),h0_lambda_mat=np.identity(2)*3)
    # # print(learn_model.get_h0_params())
    # # learn_model.save_h0_params("h0_params")

    # # learn_model.set_h0_params(h0_mu_vec=np.ones(3)/2,h0_lambda_mat=np.identity(3)*5,h0_a=10,h0_b=2)
    # # print(learn_model.get_h0_params())

    # # learn_model.load_h0_params("h0_params.npz")
    # # print(learn_model.get_h0_params())

    # # length = 100
    # # gen_model = GenModel(theta_vec=np.array([0.0,1.0]),tau=10)
    # # x = gen_model.gen_sample(length)

    # # learn_model = LearnModel(degree=3)
    # # print(x[:3])
    # # learn_model.update_posterior(x[:3])

    # # plt.plot(x[1:])
    
    # # pred_values = np.zeros(length)
    # # upper_values = np.zeros(length)
    # # lower_values = np.zeros(length)
    # # learn_model = LearnModel()
    # # for i in range(1,length):
    # #     pred_values[i] = learn_model.pred_and_update(x[i-1:i+1])
    # #     lower_values[i], upper_values[i] = learn_model.predict_interval(0.9)
    # # learn_model.pred_and_update(x[:2])
    # # print(learn_model.get_hn_params())

    # # learn_model.reset_hn_params()
    # # learn_model.update_posterior(x[:2])
    # # print(learn_model.get_hn_params())

    # # plt.plot(pred_values[1:])
    # # plt.plot(lower_values[1:])
    # # plt.plot(upper_values[1:])

    # # plt.show()

    # # learn_model.visualize_posterior()
    # # print(learn_model.get_hn_params())
    # # learn_model.save_hn_params("hn_params")

    # # learn_model.reset_hn_params()
    # # print(learn_model.get_hn_params())
    # # learn_model.update_posterior(x)
    # # print(learn_model.get_hn_params())
    # # learn_model.visualize_posterior()

if __name__ == '__main__':
    gen_model = GenModel(2)
    h_params = gen_model.get_h_params()
    params = gen_model.get_params()
    learn_model = LearnModel(2)
    h0_params = learn_model.get_h0_params()
    hn_params = learn_model.get_hn_params()

    learn_model.set_h0_params(*h_params.values())
    print('ok1')
    learn_model.set_h0_params(*h0_params.values())
    print('ok2')
    learn_model.set_h0_params(*hn_params.values())
    print('ok3')
    try:
        learn_model.set_h0_params(**params)
        print('!!!!!!!!!!!NG!!!!!!!!!!!!')
    except:
        print('ok4')

    learn_model.set_hn_params(*h_params.values())
    print('ok5')
    learn_model.set_hn_params(*h0_params.values())
    print('ok6')
    learn_model.set_hn_params(*hn_params.values())
    print('ok7')
    try:
        learn_model.set_hn_params(**params)
        print('!!!!!!!!!!!NG!!!!!!!!!!!!')
    except:
        print('ok8')

    gen_model.set_h_params(*h_params.values())
    print('ok9')
    gen_model.set_h_params(*h0_params.values())
    print('ok10')
    gen_model.set_h_params(*hn_params.values())
    print('ok11')
    try:
        gen_model.set_h_params(**params)
        print('!!!!!!!!!!!NG!!!!!!!!!!!!')
    except:
        print('ok12')

    gen_model.set_params(**params)
    print('ok13')
    try:
        gen_model.set_params(*h_params.values())
        print('!!!!!!!!!!!NG!!!!!!!!!!!!')
    except:
        print('ok14')

    import copy
    gen_model = GenModel(2)
    x = gen_model.gen_sample(100)
    learn_model = LearnModel(2)
    h0_params = copy.deepcopy(learn_model.get_h0_params())
    hn_params = copy.deepcopy(learn_model.get_hn_params())
    learn_model.update_posterior(x)
    if str(hn_params) != str(learn_model.get_hn_params()):
        print('ok15')
    else:
        print('!!!!!!!!!!!NG!!!!!!!!!!!!')
    learn_model.reset_hn_params()
    if str(hn_params) == str(learn_model.get_hn_params()):
        print('ok16')
    else:
        print('!!!!!!!!!!!NG!!!!!!!!!!!!')
    learn_model.update_posterior(x)
    learn_model.overwrite_h0_params()
    if str(h0_params) != str(learn_model.get_h0_params()):
        print('ok17')
    else:
        print('!!!!!!!!!!!NG!!!!!!!!!!!!')
