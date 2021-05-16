import numpy as np

def build_W(letra_1, letra_2, letra_3, letra_4):
    #TODO  Acá deberia ciclar por 4 letras y sacar W
    K = np.array([letra_1, letra_2, letra_3, letra_4])
    W = K.transpose().dot(K)
    W = W.dot(1/25)
    np.fill_diagonal(W, 0)
    return np.array(W)


def check_if_patron(current_state, prev_state):
    if np.array_equal(prev_state, current_state):
        return False, current_state
    return True, current_state


def next_step(state, W):
    state = np.dot(W, state)
    state = np.sign(state)
    return state


def mutate_letter(letra, prob):
    for i in range(len(letra)):
        if np.random.sample() < prob:
            letra[i] = letra[i]*(-1)
    return letra




def main():
    #TODO Guardar la letra y las letras con ruido 5x5
    letra_a = [-1, -1, 1, -1, -1 , -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1]
    letra_b = [-1, 1, 1, -1, -1, -1, 1,	-1, 1, -1, -1, 1, 1, -1, -1, -1, 1,	-1,	1, -1, -1, 1, 1, -1, -1]
    letra_c = [-1, -1, 1, 1, -1, -1, 1, -1,	-1,	-1,	-1,	1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1]
    letra_d = [-1, 1, 1, -1, -1, -1, 1, -1,	1, -1, -1, 1, -1, 1, -1, -1, 1, -1,	1, -1, -1, 1, 1, -1, -1]
    letra_e = [-1, 1, 1, 1, -1,	-1,	1, -1, -1, -1, -1, 1, 1, 1, -1,	-1,	1, -1, -1, -1, -1, 1, 1, 1,	-1]
    letra_f = [-1, 1, 1, 1,	-1,	-1,	1, -1, -1, -1, -1, 1, 1, 1,	-1,	-1,	1, -1, -1, -1, -1, 1, -1, -1, -1]
    letra_g = [-1,	-1,	1,	1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	1,	-1,	-1]
    letra_h = [-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1]
    letra_i = [-1,	1,	1,	1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	1,	1,	1,	-1]
    letra_j = [-1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1]
    letra_k = [-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1]
    letra_l = [-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	-1]
    letra_m = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1]
    letra_n = [-1,	1, 1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1]
    letra_o = [-1,	-1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	1,	-1,	-1]
    letra_p = [-1,	1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1]
    letra_q = [-1,	-1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	1,	1	,1]
    letra_r = [-1,	1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1]
    letra_s = [-1,	-1,	1,	1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1]
    letra_t = [-1,	1,	1,	1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1]
    letra_u = [-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1	,1,	1	,-1]
    letra_v = [-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1	,1,	-1,	-1]
    letra_w = [1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1	,-1,	1,	-1]
    letra_x = [-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	1	,-1,	1,	-1]
    letra_y = [-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	1	,1,	1,	-1]
    letra_z = [-1,	1,	1, 1, -1, -1,	-1,	-1,	1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	-1]


    # A = a25, b17, c11, d17, e15, f15, g13, h3, i9, j17, k11, l19, m21, n17, o17, p17, q21, r11,s , t1, u15, v11, w7 ,x15, y13, z11 ==> COMPLETO {H, T}

    # E = a3, b7, c9, d3, e13, f9, g3, h5, i25, j11, k7, l9, m3, n3, o1, p7, q1, r7, s13, t21, u5, v1, w3 ,x9, y11, z17 ===> CHOTO

    # i = a3, b7, c9, d3, e13, f9, g3, h5, i25, j11, k7, l9, m3, n3, o1, p7, q1, r7, s13, t21, u5, v1, w3 ,x9, y11, z17 ==> COMPLETO {

    # V = a3, b7, c9, d3, e13, f9, g3, h5, i25, j11, k7, l9, m3, n3, o1, p7, q1, r7, s13, t21, u5, v1, w3 ,x9, y11, z17 ===> CHOTO

    # O = a3, b7, c9, d3, e13, f9, g3, h5, i25, j11, k7, l9, m3, n3, o1, p7, q1, r7, s13, T3, u5, v1, w3 ,x9, y11, z17  ==> CHOTO {T3}

    # T = a1, b7, c9, d3, e19, f9, g7, h1, i21, j7, k3, l5, m7, n1, o3, p7, q1, r3, s13, T25, u1, v5, w1 ,x5, y7, z13  ==> COMPLETO
        # { A D H K N O Q R U W }

    # H => CHOTO EXP {T}

    # K => CHOTO ECT {T}

    # N => h3, t1, s7

    # R => f9, i9, h7, s3, w7,

    # S => g9, k9, l5, m7m o7

    #ma19, mb11, mc9, md15, me9, mf9, mg11, mh13,mi3 ,mj11, mk11, mm25, mn19, ml9, mo15 , mp11, mq15, mr15, ms5, mtt, mu, mv, mw5, mx, my, mz

    # print(np.dot(letra_j, letra_k))
    # print(np.dot(letra_j, letra_o))
    # print(np.dot(letra_k, letra_o))
    # print(np.dot(letra_k, letra_i))
    # print(np.dot(letra_i, letra_o))
    # print(np.dot(letra_j, letra_i))
    # print('000')
    # print(np.dot(letra_j, letra_h))
    # print(np.dot(letra_j, letra_i))
    # print(np.dot(letra_j, letra_j))
    # print(np.dot(letra_j, letra_k))
    # print(np.dot(letra_j, letra_l))
    # print(np.dot(letra_j, letra_m))
    # print(np.dot(letra_j, letra_n))
    # print(np.dot(letra_j, letra_o))
    # print(np.dot(letra_j, letra_p))
    # print(np.dot(letra_j, letra_q))
    # print(np.dot(letra_j, letra_r))
    # print(np.dot(letra_j, letra_s))
    # print(np.dot(letra_j, letra_t))
    # print(np.dot(letra_j, letra_u))
    # print(np.dot(letra_j, letra_v))
    # print(np.dot(letra_j, letra_w))
    # print(np.dot(letra_j, letra_x))
    # print(np.dot(letra_j, letra_y))
    # print(np.dot(letra_j, letra_z))

    # 1 = {io, iv, iq
    # 2 = {
    # 3 = {ia, id, ig, im, in, is,
    # 4 = {
    # 9 = {o-k




    current_state = np.array(mutate_letter(letra_b, 0.2))
    prev_state = np.array([0])
    #TODO Funcion para crear W : recibe la letra sin ruido : se podrá recibir mas de una?
    W = build_W(letra_j, letra_k, letra_i, letra_o)
    condition = True
    limit = 1000
    while condition and limit:
        condition, prev_state = check_if_patron(current_state, prev_state)
        current_state = next_step(current_state, W)
        limit -= 1

    if np.array_equal(letra_j, current_state):
        print('Encontrado J')
    elif np.array_equal(letra_b, current_state):
        print('Encontrado B')
    elif np.array_equal(letra_a, current_state):
        print('Encontrado A')
    elif np.array_equal(letra_m, current_state):
        print('Encontrado M')
    elif np.array_equal(letra_k, current_state):
        print('Encontrado K')
    elif np.array_equal(letra_o, current_state):
        print('Encontrado O')
    elif np.array_equal(letra_i, current_state):
        print('Encontrado I')
    else: print('No encontrado')






# call main
if __name__ == '__main__':
    main()



#########################################‹CEMENTERIO>#######################################
# #Letras en vectores
# letra_jota = [[-1, 1, 1, 1, 1], [-1, -1, -1, 1, -1], [-1, -1, -1, 1, -1], [-1, -1, -1, 1, -1], [-1, 1, 1, 1, -1]]
# letra_jota_poco_ruido = [[-1, 1, 1, 1, 1], [-1, -1, -1, 1, -1], [-1, -1, -1, 1, 1], [1, -1, -1, 1, -1], [-1, 1, 1, 1, -1]]
# letra_jota_mucho_ruido = [[1, -1, -1, -1, 1], [-1, 1, 1, 1, -1], [-1, -1, -1, 1, 1], [1, -1, 1, 1, -1], [-1, 1, -1, 1, -1]]

#   -a -b -c -d -e -f -g -h -i -j -k -l -m -n -o -p -q -r -s -t -u -v -w -x -y  -yz
# a
# b
# c 11 25 11 17 13 19 09 09 07 11 17 -9 11 15 11 15 11 13 09 13 13 -9 09 07 13
# d
# e
# f
# g 13 19 17 15 15 25 11 03 09 09 11 -11 13 21 09 17 09 15 07 15 19 -15 07 09 07
# h
# i
# j 09 07 13 11 07 09 11 11 25 09 07 -11 13 09 05 09 09 11 7 15 11 -7 11 17 15
# k
# l
# m
# n 21 11 21 15 15 13 19 03 13 17 15 -19 25 17 17 17 21 7 -1 19 15 -11 15 13 11
# o
# p
# q
# r
# s 11 13 11 13 13 15 09 13 11 11 09 -5 07 11 11 07 11 25 13 09 09 -5 13 15 13
# t -1 09 03 09 09 07 01 21 07 03 05 07 -1 03 07 -1 03 13 25 01 05 -1 05 07 13
# u
# v
# w
# x
# y
# z


