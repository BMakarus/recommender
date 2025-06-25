import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer

ratings = np.genfromtxt("C:/Users/Użytkownik/PycharmProjects/project1_s332309/data/ratings.csv", delimiter=",", skip_header=1)

movieIds = np.unique(ratings[:, 1])
n, d = len(np.unique(ratings[:, 0])), len(movieIds)
movieIds_dict = dict([(movieIds[k], k) for k in range(d)])

Z_true = np.full((n, d), np.nan)
for k in range(ratings.shape[0]):
    userId, movieId, rating = ratings[k, :3]
    Z_true[int(userId)-1, movieIds_dict[movieId]] = float(rating)
Z_true = np.ma.masked_invalid(Z_true)
Z_zeroes = Z_true.filled(fill_value=0)

lambda_grid = {"alpha": [k/10 for k in range(1, 11)]}
lambdas_opt = []
for k in range(d):
    Z_fit = Z_zeroes[np.where(Z_zeroes[:, k] != 0)]
    L = Lasso(max_iter=1000)
    if Z_fit.shape[0] > 1:
        CV = GridSearchCV(L, lambda_grid, cv=min(5, Z_fit.shape[0]), scoring="neg_mean_squared_error")
        CV.fit(np.hstack((Z_fit[:, :k], Z_fit[:, (k+1):])), Z_fit[:, k])
        lambdas_opt.append(CV.best_params_["alpha"])
    else:
        lambdas_opt.append(lambdas_opt[-1])
    print(k*100/d)

x_bar = np.nanmean(Z, axis=0)
Z_means = np.copy(Z)
Z_means[np.where(Z.mask)] = np.tile(x_bar, n).reshape((n, d))[np.where(Z.mask)]



Z_fit = Z_zeroes[np.where(Z_zeroes[:, k] != 0)]
L = Lasso(max_iter=1000)
if Z_fit.shape[0] > 1:
    CV = GridSearchCV(L, lambda_grid, cv=min(5, Z_fit.shape[0]), scoring="neg_mean_squared_error")
    CV.fit(np.hstack((Z_fit[:, :k], Z_fit[:, (k+1):])), Z_fit[:, k])
   lambdas_opt.append(CV.best_params_["alpha"])
else:
    lambdas_opt.append(lambdas_opt[-1])
    print(k*100/d)

k = 2023
MSEs=[]
for a in np.arange(0.01, 2, 0.01):
    L = Lasso(alpha=a, max_iter=10000)
    Z_fit = Z_zeroes[np.where(Z_zeroes[:, k] != 0)]
    CV = KFold(n_splits=min(5, Z_fit.shape[0]), shuffle=True)
    CV_scores = cross_val_score(L, np.hstack((Z_fit[:, :k], Z_fit[:, (k+1):])), Z_fit[:, k], cv=CV, scoring='neg_mean_squared_error')
    MSEs.append(-CV_scores.mean())
    print(a, "MSE=", -CV_scores.mean())

plt.plot(np.arange(0.01, 2, 0.01), MSEs)

lam=0.25

for k in range(d):
    L = Lasso(alpha=0.25, max_iter=10000)
    Z_fit, Z_predict = Z_zeroes[np.where(Z_zeroes[:, k] != 0)], Z_zeroes[np.where(Z_zeroes[:, k] == 0)]
    L.fit(np.hstack((Z_fit[:, :k], Z_fit[:, (k+1):])), Z_fit[:, k])
    Z[:, k][np.where(Z.mask[:, k])] = L.predict(np.hstack((Z_predict[:, :k], Z_predict[:, (k+1):])))


#imp

mean_errors = []

for n_neighbors in range(1, 26):
    errors = []
    k=0
    while k < 5:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        indices = np.random.choice(100836, size=5, replace=False)
        i, j = np.where(Z_true.mask == False)[0][indices], np.where(Z_true.mask == False)[1][indices]
        Z_train = np.copy(Z_true.data)
        Z_train[i, j] = np.nan
        Z_imp = imputer.fit_transform(Z_train)
        if Z_imp.shape[1] == 9724:
            errors.append(np.sum((Z_true.data[i, j]-Z_imp[i, j])**2)/len(i))
            k+=1
        else:
            continue
    mean_errors.append(np.mean(errors))
    print("n=", n_neighbors, "mean_error=", np.mean(errors))

# refugees from helper

Z_unimp = np.copy(Z.data)

Z, user_map, movie_map = build_rating_matrix("C:/Users/Użytkownik/PycharmProjects/project1_s332309/data/ratings.csv")


# NMF
RMSEs = []
for n_components in range(1, 51):

    model = NMF(n_components=n_components, init='random')
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)

    RMSE = np.sqrt(np.sum((Z_zeroes[np.where(Z_zeroes != 0)]-Z_approx[np.where(Z_zeroes != 0)])**2)/100836)
    print("r=", n_components, "RMSE=", RMSE)
    RMSEs.append(RMSE)

plt.plot(np.arange(1, 51), RMSEs)

# NMF all obs
RMSEs = []
for n_components in np.arange(1, 5001, 500):

    model = NMF(n_components=n_components, init='random')
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)

    RMSE = np.sqrt(np.sum((Z-Z_approx)**2)/(610*9724))
    print("r=", n_components, "RMSE=", RMSE)
    RMSEs.append(RMSE)

# SVD
RMSEs = []
for n_components in np.arange(1, 5000, 100):

    model = TruncatedSVD(n_components=n_components)
    model.fit(Z)
    W = model.transform(Z) / model.singular_values_
    Lambda = np.diag(model.singular_values_)
    V_T = model.components_
    H = np.dot(Lambda, V_T)
    Z_approx = np.dot(W, H)

    RMSE = np.sqrt(np.sum((Z_zeroes[np.where(Z_zeroes != 0)]-Z_approx[np.where(Z_zeroes != 0)])**2)/100836)
    print("r=", n_components, "RMSE=", RMSE)
    RMSEs.append(RMSE)

plt.plot(np.arange(1, 5000, 100), RMSEs)

# SVD all obs
RMSEs = []
for n_components in np.arange(1, 5000, 100):

    model = TruncatedSVD(n_components=n_components)
    model.fit(Z)
    W = model.transform(Z) / model.singular_values_
    Lambda = np.diag(model.singular_values_)
    V_T = model.components_
    H = np.dot(Lambda, V_T)
    Z_approx = np.dot(W, H)

    RMSE = np.sqrt(np.sum((Z-Z_approx)**2)/(610*9724))
    print("r=", n_components, "RMSE=", RMSE)
    RMSEs.append(RMSE)

#SGD

Z_mask_torch = torch.from_numpy(Z.mask)

Z_torch = torch.from_numpy(Z)

lr = 0.00001
n_epochs = 1000

for r in [1250, 1500, 1750]:

    W = torch.randn((610, r), requires_grad=True, dtype=torch.float, device="cpu")
    H = torch.randn((r, 9724), requires_grad=True, dtype=torch.float, device="cpu")


    optimizer = torch.optim.SGD([W, H], lr=lr)

    loss_list = []

    for epoch in range(n_epochs):
        loss = torch.sum((Z_torch[torch.where(Z_mask_torch == False)]-torch.matmul(W, H)[torch.where(Z_mask_torch == False)])**2)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())
    print("r=", r, "loss=", loss_list[-1])

n_components=2

ratings = np.genfromtxt("C:/Users/Użytkownik/PycharmProjects/project1_s332309/data/ratings.csv", delimiter=',', skip_header=1)

user_ids, movie_ids = np.unique(ratings[:, 0]), np.unique(ratings[:, 1])

user_map, movie_map = {user_id: i for i, user_id in enumerate(sorted(user_ids))}, \
                          {movie_id: j for j, movie_id in enumerate(sorted(movie_ids))}

n, d = len(user_ids), len(movie_ids)

Z = np.full((n, d), np.nan)
for i in range(ratings.shape[0]):
    user_id, movie_id, rating = ratings[i, :3]
    Z[user_map[user_id], movie_map[movie_id]] = rating
Z = np.ma.masked_invalid(Z)
Z_mask_torch = torch.from_numpy(Z.mask)
Z_torch = torch.from_numpy(Z.data)

lr = 0.0001
n_epochs = 1000

W = torch.randn((n, n_components), requires_grad=True, dtype=torch.float, device="cpu")
H = torch.randn((n_components, d), requires_grad=True, dtype=torch.float, device="cpu")

optimizer = torch.optim.SGD([W, H], lr=lr)

loss_list = []

for epoch in range(n_epochs):
        loss = torch.sum((Z_torch[torch.where(Z_mask_torch == False)] - torch.matmul(torch.exp(W), torch.exp(H))[torch.where(Z_mask_torch == False)]) ** 2)+torch.sum(torch.exp(W))+torch.sum(torch.exp(H))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())
        print("loss=", loss_list[-1])

plt.plot(np.arange(0, 1000), loss_list)

def RMSE(Z_true, Approx):
    return np.sqrt(np.sum((Z_true[np.where(Z_true.mask == False)] - Approx[np.where(Z_true.mask == False)]) ** 2)/100836)

#RMSEs_sdg = []
#for r in np.arange(1, 50, 5):
#    W, H, user_map, movie_map = train_sgd_model("C:/Users/Użytkownik/PycharmProjects/project1_s332309/data/ratings.csv", n_components=r, batch_size=10000, n_epochs=2000)
#    RMSEs_sdg.append(RMSE(Z_true, torch.matmul(W, H).detach().numpy()))
#    print("r=", r, "RMSE=", RMSE(Z_true, torch.matmul(W, H).detach().numpy()))

#RMSEs_nmf = []
#for r in np.arange(1, 501, 100):
#    Z, user_map, movie_map = train_nmf_model("C:/Users/Użytkownik/PycharmProjects/project1_s332309/data/ratings.csv", n_components=r)
#    RMSEs_nmf.append(RMSE(Z_true, Z))
#    print("r=", r, "RMSE=", RMSE(Z_true, Z))

#RMSEs_svd = []
#for r in np.arange(1, 501, 100):
#    Z, user_map, movie_map = train_svd_model("C:/Users/Użytkownik/PycharmProjects/project1_s332309/data/ratings.csv", n_components=r)
#    RMSEs_svd.append(RMSE(Z_true, Z))
#    print("r=", r, "RMSE=", RMSE(Z_true, Z))

#RMSEs_svd2 = []
#for r in np.arange(1, 501, 100):
#    Z, user_map, movie_map = train_svd_model("C:/Users/Użytkownik/PycharmProjects/project1_s332309/data/ratings.csv", n_components=r)
#    RMSEs_svd2.append(RMSE(Z_true, Z))
#    print("r=", r, "RMSE=", RMSE(Z_true, Z))