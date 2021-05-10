import numpy as np
import random
import time
import argparse

# random.seed(123)
# ml 100k
usize = 16239 + 1
msize = 14284 + 1


# tsize = 18 + 1


# ml 1m
# usize = 6040 + 1
# msize = 3706 + 1

def parse_args():
    parser = argparse.ArgumentParser(description="Run MCRec.")
    parser.add_argument('--walk_num', type=int, default=5,
                        help='the length of random walk .')
    parser.add_argument('--metapath', type=str, default="ubub",
                        help='the metapath for yelp dataset. Recommend: UBUB, UBCaB, UUB, UBCiB')
    return parser.parse_args()


class MetapathBasePathSample:
    def __init__(self, **kargs):
        self.metapath = kargs.get('metapath')
        self.walk_num = kargs.get('walk_num')
        self.K = kargs.get('K')
        self.ub_dict = dict()
        self.bu_dict = dict()
        self.bca_dict = dict()
        self.cab_dict = dict()
        self.uco_dict = dict()
        self.cou_dict = dict()
        self.bci_dict = dict()
        self.cib_dict = dict()
        self.uu_dict = dict()
        # self.mm_dict = dict()
        # self.um_list = list()

        self.user_embedding = np.random.rand(usize, 64)
        self.item_embedding = np.random.rand(msize, 64)
        # self.type_embedding = np.random.rand(tsize, 64)
        print('Begin to load data')
        start = time.time()

        # self.load_user_embedding('../data/ml-100k.bpr.user_embedding')
        # self.load_item_embedding('../data/ml-100k.bpr.item_embedding')
        # self.load_type_embedding('../data/ml-100k.bpr.type_embedding')

        self.load_ub(kargs.get('ubfile'))
        self.load_bca(kargs.get('bcafile'))
        self.load_uu(kargs.get('uufile'))
        self.load_bci(kargs.get('bcifile'))
        self.load_uco(kargs.get('ucofile'))
        end = time.time()
        print('Load data finished, used time %.2fs' % (end - start))
        self.path_list = list()
        self.outfile = open(kargs.get('outfile_name'), 'w')
        self.metapath_based_randomwalk()
        self.outfile.close()

    # def load_user_embedding(self, ufile):
    #     with open(ufile) as infile:
    #         for line in infile.readlines():
    #             arr = line.strip().split(' ')
    #             i = int(arr[0])
    #             for j in range(len(arr[1:])):
    #                 self.user_embedding[i][j] = float(arr[j + 1])
    #
    # def load_item_embedding(self, ifile):
    #     with open(ifile) as infile:
    #         for line in infile.readlines():
    #             arr = line.strip().split(' ')
    #             i = int(arr[0])
    #             for j in range(len(arr[1:])):
    #                 self.item_embedding[i][j] = float(arr[j + 1])
    #
    # def load_type_embedding(self, tfile):
    #     with open(tfile) as infile:
    #         for line in infile.readlines():
    #             arr = line.strip().split(' ')
    #             i = int(arr[0])
    #             for j in range(len(arr[1:])):
    #                 self.type_embedding[i][j] = float(arr[j + 1])

    def metapath_based_randomwalk(self):
        pair_list = []
        for u in range(1, usize):
            for i in range(1, msize):
                pair_list.append([u, i])
        print('load pairs finished num = ', len(pair_list))
        ctn = 0
        t1 = time.time()
        avg = 0
        print(len(pair_list))
        for u, b in pair_list:
            ctn += 1
            # print u, m
            if ctn % 10000 == 0:
                print('%d [%.4f]\n' % (ctn, time.time() - t1))
            if self.metapath == 'ubcib':
                self.walk_ubcib(u, b)
            elif self.metapath == 'ubcab':
                self.walk_ubcab(u, b)
            elif self.metapath == 'ubub':
                self.walk_ubub(u, b)
            elif self.metapath == 'uub':
                self.walk_uub(u, b)
            else:
                print('unknow metapath.')
                exit(0)

    def get_sim(self, u, v):
        return u.dot(v) / ((u.dot(u) ** 0.5) * (v.dot(v) ** 0.5))

    def walk_ubub(self, s_u, e_b):
        limit = 10
        b_list = []
        for b in self.ub_dict[s_u]:
            sim = self.get_sim(self.user_embedding[s_u],
                               self.item_embedding[b])  # self.user_embedding[s_u].dot(self.item_embedding[m]) /
            b_list.append([b, sim])
        b_list.sort(key=lambda x: x[1], reverse=True)
        b_list = b_list[:min(limit, len(b_list))]

        u_list = []
        for u in self.bu_dict.get(e_b, []):
            sim = self.get_sim(self.item_embedding[e_b],
                               self.user_embedding[u])  # self.item_embedding[e_b].dot(self.user_embedding[u])
            u_list.append([u, sim])
        u_list.sort(key=lambda x: x[1], reverse=True)
        u_list = u_list[:min(limit, len(u_list))]

        bu_list = []
        for b in b_list:
            for u in u_list:
                bb = b[0]
                uu = u[0]
                if bb in self.bu_dict and uu in self.bu_dict[bb] and uu != s_u and bb != e_b:
                    sim = (self.get_sim(self.user_embedding[uu], self.item_embedding[bb]) + u[1] + b[1]) / 3.0
                    if sim > 0.7:
                        bu_list.append([bb, uu, sim])
        bu_list.sort(key=lambda x: x[2], reverse=True)
        bu_list = bu_list[:min(5, len(bu_list))]

        if (len(bu_list) == 0):
            return
        self.outfile.write(str(s_u) + ',' + str(e_b) + '\t' + str(len(bu_list)))
        for bu in bu_list:
            path = ['u' + str(s_u), 'b' + str(bu[0]), 'u' + str(bu[1]), 'b' + str(e_b)]
            self.outfile.write('\t' + '-'.join(path) + ' ' + str(bu[2]))
        self.outfile.write('\n')

    def walk_ubcab(self, s_u, e_b):
        limit = 10
        b_list = []
        for b in self.ub_dict[s_u]:
            sim = self.get_sim(self.user_embedding[s_u], self.item_embedding[b])
            b_list.append([b, sim])
        b_list.sort(key=lambda x: x[1], reverse=True)
        b_list = b_list[:min(limit, len(b_list))]

        ca_list = []
        for ca in self.bca_dict.get(e_b, []):
            ca_list.append([ca, 1])

        bca_list = []
        for b in b_list:
            for ca in ca_list:
                bb = b[0]
                caca = ca[0]
                if bb in self.bca_dict and caca in self.bca_dict[bb] and bb != e_b:
                    sim = b[1]
                    if sim > 0.7:
                        bca_list.append([bb, caca, sim])
        bca_list.sort(key=lambda x: x[2], reverse=True)
        bca_list = bca_list[:min(5, len(bca_list))]

        if (len(bca_list) == 0):
            return
        self.outfile.write(str(s_u) + ',' + str(e_b) + '\t' + str(len(bca_list)))
        for mt in bca_list:
            path = ['u' + str(s_u), 'b' + str(mt[0]), 'ca' + str(mt[1]), 'b' + str(e_b)]
            self.outfile.write('\t' + '-'.join(path))
        self.outfile.write('\n')

    def walk_ubcib(self, s_u, e_b):
        limit = 10
        b_list = []
        for b in self.ub_dict[s_u]:
            sim = self.get_sim(self.user_embedding[s_u], self.item_embedding[b])
            b_list.append([b, sim])
        b_list.sort(key=lambda x: x[1], reverse=True)
        b_list = b_list[:min(limit, len(b_list))]

        ci_list = []
        for ci in self.bci_dict.get(e_b, []):
            ci_list.append([ci, 1])

        bci_list = []
        for b in b_list:
            for ci in ci_list:
                bb = b[0]
                cici = ci[0]
                if bb in self.bci_dict and cici in self.bci_dict[bb] and bb != e_b:
                    sim = b[1]
                    if sim > 0.7:
                        bci_list.append([bb, cici, sim])
        bci_list.sort(key=lambda x: x[2], reverse=True)
        bci_list = bci_list[:min(5, len(bci_list))]

        if (len(bci_list) == 0):
            return
        self.outfile.write(str(s_u) + ',' + str(e_b) + '\t' + str(len(bci_list)))
        for mt in bci_list:
            path = ['u' + str(s_u), 'b' + str(mt[0]), 'ci' + str(mt[1]), 'b' + str(e_b)]
            self.outfile.write('\t' + '-'.join(path))
        self.outfile.write('\n')

    def walk_uub(self, s_u, e_b):
        limit = 10

        us_list = []
        for us in self.bu_dict.get(e_b, []):
            sim = self.get_sim(self.item_embedding[e_b], self.user_embedding[us])
            us_list.append([us, sim])
        us_list.sort(key=lambda x: x[1], reverse=True)
        us_list = us_list[:limit]

        u_list = []
        for us in us_list:
            uss = us[0]
            if s_u in self.uu_dict and uss in self.uu_dict[s_u] and uss != s_u:
                sim = us[1]
                if sim > 0.7:
                    u_list.append([uss, sim])
        u_list.sort(key=lambda x: x[1], reverse=True)
        u_list = u_list[:5]

        if (len(u_list) == 0):
            return
        self.outfile.write(str(s_u) + ',' + str(e_b) + '\t' + str(len(u_list)))
        for uu in u_list:
            path = ['u' + str(s_u), 'u' + str(uu[0]), 'b' + str(e_b)]
            self.outfile.write('\t' + '-'.join(path))
        self.outfile.write('\n')
    #
    # def walk_ummm(self, s_u, e_b):
    #     limit = 10
    #     mf_list = []
    #     for mf in self.ub_dict[s_u]:
    #         sim = self.get_sim(self.item_embedding[mf], self.user_embedding[s_u])
    #         mf_list.append([mf, sim])
    #     mf_list.sort(key=lambda x: x[1], reverse=True)
    #     mf_list = mf_list[:limit]
    #
    #     ms_list = []
    #     for ms in self.mm_dict.get(e_b, []):
    #         ms_list.append([ms, 1])
    #
    #     mm_list = []
    #     for mf in mf_list:
    #         for ms in ms_list:
    #             mff = mf[0]
    #             mss = ms[0]
    #             if mff in self.mm_dict and mss in self.mm_dict[mff] and mff != e_b:
    #                 sim = mf[1]
    #                 if sim > 0.7:
    #                     mm_list.append([mff, mss, sim])
    #     mm_list.sort(key=lambda x: x[2], reverse=True)
    #     mm_list = mm_list[:5]
    #
    #     if (len(mm_list) == 0):
    #         return
    #     self.outfile.write(str(s_u) + ',' + str(e_b) + '\t' + str(len(mm_list)))
    #     for mm in mm_list:
    #         path = ['u' + str(s_u), 'm' + str(mm[0]), 'm' + str(mm[1]), 'm' + str(e_b)]
    #         self.outfile.write('\t' + '-'.join(path) + ' ' + str(mm[2]))
    #     self.outfile.write('\n')
    #
    # def walk_mumt(self, start, end):
    #     path = ['m' + str(start)]
    #
    #     # m - u
    #     # print start
    #     if start not in self.bu_dict:
    #         return None
    #     index = np.random.randint(len(self.bu_dict[start]))
    #     u = self.bu_dict[start][index]
    #     path.append('u' + str(u))
    #     # u - m
    #     if u not in self.ub_dict:
    #         return None
    #     index = np.random.randint(len(self.ub_dict[u]))
    #     m = self.ub_dict[u][index]
    #     path.append('m' + str(m))
    #     # m - t
    #     # print path
    #     if m not in self.bca_dict:
    #         return None
    #     if end not in self.bca_dict[m]:
    #         return None
    #     path.append('t' + str(end))
    #     return '-'.join(path)
    #
    # def walk_mumumt(self, start, end):
    #     path = ['m' + str(start)]
    #
    #     # m - u
    #     # print start
    #     if start not in self.bu_dict:
    #         return None
    #     index = np.random.randint(len(self.bu_dict[start]))
    #     u = self.bu_dict[start][index]
    #     path.append('u' + str(u))
    #     # u - m
    #     if u not in self.ub_dict:
    #         return None
    #     index = np.random.randint(len(self.ub_dict[u]))
    #     m = self.ub_dict[u][index]
    #     path.append('m' + str(m))
    #
    #     # m - u
    #     if m not in self.bu_dict:
    #         return None
    #     index = np.random.randint(len(self.bu_dict[m]))
    #     u = self.bu_dict[m][index]
    #     path.append('u' + str(u))
    #
    #     # u - m
    #     if u not in self.ub_dict:
    #         return None
    #     index = np.random.randint(len(self.ub_dict[u]))
    #     m = self.ub_dict[u][index]
    #     path.append('m' + str(m))
    #
    #     # m - t
    #     # print path
    #     if m not in self.bca_dict:
    #         return None
    #     if end not in self.bca_dict[m]:
    #         return None
    #     path.append('t' + str(end))
    #     return '-'.join(path)

    def random_walk(self, start):
        path = [self.metapath[0] + start]
        iterator = 0
        k = 1
        while True:
            if k == len(self.metapath):
                iterator += 1
                k = 0
                if iterator == K:
                    return '-'.join(path)

            if k == 0 and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.bu_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif k == 0 and self.metapath[k] == 'm':
                pre = path[-1][1:]
                neighbors = self.ub_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k - 1] == 'u' and self.metapath[k] == 'm':
                pre = path[-1][1:]
                neighbors = self.ub_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k - 1] == 'm' and self.metapath[k] == 't':
                pre = path[-1][1:]
                neighbors = self.bca_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k - 1] == 't' and self.metapath[k] == 'm':
                pre = path[-1][1:]
                neighbors = self.cab_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k - 1] == 'm' and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.bu_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k - 1] == 'u' and self.metapath[k] == 'a':
                pre = path[-1][1:]
                neighbors = self.bci_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
            elif self.metapath[k - 1] == 'a' and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.cib_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k - 1] == 'u' and self.metapath[k] == 'o':
                pre = path[-1][1:]
                neighbors = self.uco_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
            elif self.metapath[k - 1] == 'o' and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.cou_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

    def load_ub(self, ubfile):
        with open(ubfile) as infile:
            for line in infile.readlines():
                u, b = line.strip().split('\t')[:2]
                u, b = int(u), int(b)
                # self.um_list.append([u, m]);
                if u not in self.ub_dict:
                    self.ub_dict[u] = list()
                self.ub_dict[u].append(b)

                if b not in self.bu_dict:
                    self.bu_dict[b] = list()
                self.bu_dict[b].append(u)

    def load_uu(self, uufile):
        with open(uufile) as infile:
            for line in infile.readlines():
                u1, u2 = line.strip().split('\t')[:2]
                u1, u2 = int(u1), int(u2)
                if u1 not in self.uu_dict:
                    self.uu_dict[u1] = list()
                self.uu_dict[u1].append(u2)

                if u2 not in self.uu_dict:
                    self.uu_dict[u2] = list()
                self.uu_dict[u2].append(u1)

    def load_uco(self, ucofile):
        with open(ucofile) as infile:
            for line in infile.readlines():
                u, co = line.strip().split('\t')[:2]
                u, co = int(u), int(co)
                if u not in self.uco_dict:
                    self.uco_dict[u] = list()
                self.uco_dict[u].append(co)

                if co not in self.cou_dict:
                    self.cou_dict[co] = list()
                self.cou_dict[co].append(u)

    def load_bca(self, bcafile):
        with open(bcafile) as infile:
            for line in infile.readlines():
                b, ca = line.strip().split('\t')[:2]
                b, ca = int(b), int(ca)
                if b not in self.bca_dict:
                    self.bca_dict[b] = list()
                self.bca_dict[b].append(ca)

                if ca not in self.cab_dict:
                    self.cab_dict[ca] = list()
                self.cab_dict[ca].append(b)

    def load_bci(self, bcifile):
        with open(bcifile) as infile:
            for line in infile.readlines():
                b, ci = line.strip().split('\t')[:2]
                b, ci = int(b), int(ci)
                if b not in self.bci_dict:
                    self.bci_dict[b] = list()
                self.bci_dict[b].append(ci)

                if ci not in self.cib_dict:
                    self.cib_dict[ci] = list()
                self.cib_dict[ci].append(b)


if __name__ == '__main__':
    ubfile = '../data/yelp/ub_0.8.train'
    bcafile = '../data/yelp/bca.txt'
    bcifile = '../data/yelp/bci.txt'
    ucofile = '../data/yelp/uco.txt'
    uufile = '../data/yelp/uu.txt'
    # walk_num = 5
    # metapath = 'umtm'

    args = parse_args()
    walk_num = args.walk_num
    metapath = args.metapath
    K = 1

    # print ("walk_num : ", walk_num, "T : ", type(walk_num))
    # print ("meta : ", metapath, "T : ", type(metapath))
    outfile_name = '../data/yelp/yelp.' + metapath + '_' + str(walk_num) + '_' + str(K)
    print('outfile name = ', outfile_name)
    MetapathBasePathSample(uufile=uufile, bcafile=bcafile, ubfile=ubfile, bcifile=bcifile, ucofile=ucofile,
                           K=K, walk_num=walk_num, metapath=metapath, outfile_name=outfile_name)
