from re import template
import sys
import numpy as np
sys.path.append("..")
from itertools import chain
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from pattern.en import conjugate, lemma, lexeme, PRESENT, SG, PL, PAST, PROGRESSIVE
import torch

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')

# print "stopwords: ", STOPWORDS
# We remove 'own' and a few other form stopwords as in our domain it's not really a stop word
STOPWORDS.remove('own')
STOPWORDS.remove('does')
STOPWORDS.remove('do')
STOPWORDS.remove('doing')

def get_lemmas(phrase, pos=wn.VERB):
    return [LEMMATIZER.lemmatize(w, pos) for w in phrase.split(' ')]

def get_lemmas_no_stopwords(phrase, pos=wn.VERB):
    return set([w for w in get_lemmas(phrase, pos) if w not in STOPWORDS])

def aligned_args(q, a):
    q_arg = get_lemmas_no_stopwords(q[2], wn.NOUN)
    if q_arg == get_lemmas_no_stopwords(a[2], wn.NOUN):
        return True
    if q_arg == get_lemmas_no_stopwords(a[0], wn.NOUN):
        return False
    return -1

def aligned_args_rel(q, a):
    # These are not necessary if the sentences are well formed!
    q1 = LEMMATIZER.lemmatize(q[1].split("::")[0].lower())
    q2 = LEMMATIZER.lemmatize(q[2].split("::")[0].lower())
    a1 = LEMMATIZER.lemmatize(a[1].split("::")[0].lower())
    a2 = LEMMATIZER.lemmatize(a[2].split("::")[0].lower())

    if q1 == a1:
        return True
    elif q1 == a2:
        return False
    else:
        if q2 == a1:
            return False
        elif q2 == a2:
            return True
        print ("not sure if aligned: ", q, a)
        return True  # This is a bad case!

def read_data(dpath, orig_dpath,CCG,typed,LDA):
    f = open(dpath)

    if orig_dpath:
        lines_orig = open(orig_dpath).read().splitlines()
    else:
        lines_orig = None

    data = []
    idx = 0
    for l in f:
        line = l.replace("\n","")

        # if idx==10000:
        #     break

        if lines_orig:
            line_orig = lines_orig[idx]
        else:
            line_orig = None

        ss = line.split("\t")

        if len(ss)<3:
            print ("bad len problem: ", line)
            idx += 1
            continue

        q_all = ss[0].split(" ")
        p_all = ss[1].split(" ")
        q = q_all[0]
        p = p_all[0]

        if len(p_all)>1 and typed and not LDA:
            try:
                t1 = p_all[1].split("::")[1]
                t2 = p_all[2].split("::")[1]
            except:
                t1 = "thing"
                t2 = "thing"
            t1s = [t1]
            t2s = [t2]
            probs = [1]
        elif typed and LDA:
            tss = ss[3].split()
            i=0
            t1s = []
            t2s = []
            probs = []
            while i<len(tss):
                ts = tss[i].split("#")
                t1 = ts[0]
                t2 = ts[1]

                i+=1
                prob = float(tss[i])
                t1s.append(t1)
                t2s.append(t2)
                probs.append(prob)
                i+=1
        else:
            #Not well formed
            t1 = "thing"
            t2 = "thing"
            t1s = [t1]
            t2s = [t2]
            probs = [1]

        #First, let's see if the args are aligned

        if CCG:
            a = True
            if line_orig:#lazy way to check snli
                if len(q_all)>1 and len(p_all)>1:
                    a = aligned_args_rel(q_all,p_all)

        else:
            if line_orig:
                ss_orig = line_orig.split("\t")
                q_orig = ss_orig[0].split(",")
                p_orig = ss_orig[1].split(",")

                a = aligned_args([q_orig[0].strip(),"_",q_orig[2].strip()],[p_orig[0].strip(),"",p_orig[2].strip()])
                if a==-1:
                    a = aligned_args([p_orig[0].strip(),"",p_orig[2].strip()],[q_orig[0].strip(),"_",q_orig[2].strip()])
                    if a==-1:
                        raise Exception('HORRIBLE BUG!!!'+str(q)+" "+str(a))
            else:
                a = True

        try:
            q_arg1 = LEMMATIZER.lemmatize(q_all[1].split("::")[0])
            q_arg2 = LEMMATIZER.lemmatize(q_all[2].split("::")[0])

            p_arg1 = LEMMATIZER.lemmatize(p_all[1].split("::")[0])
            p_arg2 = LEMMATIZER.lemmatize(p_all[2].split("::")[0])

        except:
            print ("problem: ", line)
        #(exports.1,exports.2) nigeria oil	(supplier.of.1,supplier.of.2) nigeria oil

        if ss[2].startswith("n") or ss[2]=="False":
            l = 0
        else:
            l = 1

        data.append((p,q,t1s,t2s,probs,a,l))
        idx += 1

    return data

def min_match_length(A, B):
    for i in range(min(len(A), len(B))):
        if A[i]!=B[i]:
            return i
    return min(len(A),len(B))

type2template = {}
can_negate = ['be', 'can', 'do', 'will', 'must', 'have', 'may', 'need', 'dare', 'ought']
wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in chain(nltk.corpus.brown.tagged_words(tagset="universal"), nltk.corpus.treebank.tagged_words(tagset="universal"))) 
with open('type_map.txt', 'r') as f:
    for l in f.readlines():
        ll = l.strip().split('\t')
        type2template[ll[0]] = ll[1]

def pred2template(pred, reversing=False, negated=False):
    # pred: like (sing.2,sing.to.start.of.2)#music#event
    if pred[:5] == 'NEG__' or pred[:5] == 'neg__':
        return pred2template(pred[5:], reversing=reversing,negated=(not negated))
    try:
        preds, t1, t2 = pred.split('#')
        actor_1 = type2template[t1[:-2] if t1[-2]=='_' else t1]
        actor_2 = type2template[t2[:-2] if t2[-2]=='_' else t2]
        preds = preds[1:-1].split(',')
        preds = [p.split('.') for p in preds]
        # reversing = False
        # if self.typeList[0] == self.typeList[1]:
        #     if t1[-1] == '2' and t2[-1] == '1':
        #         reversing = True
        # else:
        #     if self.typeList[0] == t2 and self.typeList[1] == t1:
        #         reversing = True
        if reversing:
            actor_1 += ' B'
            actor_2 += ' A'
        else:
            actor_1 += ' A'
            actor_2 += ' B'
        verbFlag = ('VERB' in wordtags[preds[0][0]] or preds[0][0] == "'s")
        orig_preds = [preds[0][:-1], preds[1][:-1]]
        if orig_preds == [["'s"],["of"]]:
            prefix = "Something's " if not negated else "Not something's " 
            return prefix + actor_1 + " of " + actor_2
        if not verbFlag:
            preds[0] = ['is'] + preds[0]
            preds[1] = ['is'] + preds[1]
        # assert len(set([p[0] for p in preds])) == 1
        A_active = (preds[0][-1] == '1')
        B_active = (preds[1][-1] == '1')
        minLength = min(len(preds[0]), len(preds[1]))-1
        pathway = (preds[0][:minLength] == preds[1][:minLength])

        # the only plural case
        if A_active and B_active:
            act = False
            if preds[0] == preds[1]:
                act = preds[0][:-1]
            elif pathway and len(preds[0])-1 == minLength:
                act = preds[1][:-1][::-1]
            elif pathway and len(preds[1])-1 == minLength:
                act = preds[0][:-1][::-1]
            if act:
                if act[0] in can_negate or not negated:
                    return actor_1 + ' and ' + actor_2 + conjugate(verb=act[0],tense=PRESENT,number=PL,negated=negated) + ' ' + ' '.join(act[1:])
                return actor_1 + ' and ' + actor_2 + conjugate(verb='do',tense=PRESENT,number=PL,negated=negated) + ' ' + ' '.join(act)
            else:
                print('pathway:', pathway, preds[0], preds[1], minLength)
                raise ValueError
        if preds[0][0] not in can_negate and negated:
            preds[0] = ['do'] + preds[0]
        if preds[1][0] not in can_negate and negated:
            preds[1] = ['do'] + preds[1]
        try:
            preds[0][0] = conjugate(verb=preds[0][0],tense=PRESENT,number=SG,negated=negated)
        except Exception as e:
            print(e, 'retrying')
            preds[0][0] = conjugate(verb=preds[0][0],tense=PRESENT,number=SG,negated=negated)
        preds[1][0] = conjugate(verb=preds[1][0],tense=PRESENT,number=SG,negated=negated)
        if A_active and not B_active:
            if pathway:
                verb = ' '.join(preds[1][:-1])
                if len(preds[0]) > len(preds[1]):
                    verb = ' '.join(preds[0][:-1])
                return actor_1 + ' ' + verb + ' ' + actor_2
            else:
                mml = min_match_length(preds[0], preds[1])
                return actor_1 + ' ' + ' '.join(preds[0][:-1]) + ' Something ' + ' '.join(preds[1][mml:-1]) + actor_2
        elif B_active and not A_active:
            mml = min_match_length(orig_preds[0], orig_preds[1])
            # maybe need phrase pos-tagging?
            if 'VERB' in wordtags[orig_preds[0][0]]:
                act = orig_preds[1][::-1][:-mml] + ['to'] + orig_preds[0]
            else:
                act = orig_preds[1][::-1] + orig_preds[0][mml:]
            if 'VERB' in wordtags[act[0]]:
                if negated and act[0] not in can_negate:
                    act = ['do'] + act
                act[0] = conjugate(verb=act[0],tense=PRESENT,number=SG, negated=negated)
            else:
                if negated:
                    act = ['is', 'not'] + act
                else:
                    act = ['is'] + act
            act = [actor_1] + act + [actor_2]
            return ' '.join(act)
        elif not A_active and not B_active:
            mml = min_match_length(orig_preds[0], orig_preds[1])
            if pathway:
                if negated:
                    be = 'is not'
                else:
                    be = 'is'
                act = [actor_1, be, conjugate(verb=orig_preds[0][0],tense=PAST,aspect=PROGRESSIVE,number=SG)] + orig_preds[0][1:] + orig_preds[1][mml:] + [actor_2]
                return ' '.join(act)
            else:
                if negated:
                    if orig_preds[0] not in can_negate:
                        orig_preds[0] = ['do'] + orig_preds[0]
                    orig_preds[0][0] = conjugate(verb=orig_preds[0][0],tense=PRESENT,number=SG,negated=negated)
                act = ['Something'] + orig_preds[0] + [actor_1] + orig_preds[1][mml:] + [actor_2]
                return ' '.join(act)
    except Exception as e:
        # logging.exception(e)
        # return pred2template(pred, reversing=reversing)
        print('error',e,pred)
        return 'NULL'

    print('unprocessable pred:', pred)
    return 'NULL'

root = "../gfiles/"

temp_pairs = []
neg_temp_pairs = []

ent_rel_file = 'ent/dev_rels.txt'
with open(root + ent_rel_file, 'r') as f:
    ls = list(f.readlines())

with open('dev_corpus.txt', 'w') as f_write:
    for n, l in enumerate(ls):
        # l: (cause.1,cause.2) pharyngitis::disease fever::disease	(cause.of.1,cause.of.2) pharyngitis::disease fever::disease	True
        # ============
        reversing = False
        try:
            p, q, ans = l.rstrip().split('\t')
            p, pt1, pt2 = p.split(' ')
            p = p + '#' + pt1.split('::')[1] + '#' + pt2.split('::')[1]
            q, qt1, qt2 = q.split(' ')
            q = q + '#' + qt1.split('::')[1] + '#' + qt2.split('::')[1]
            # pp, pt1, pt2 = p.split('#')
            # qq, qt1, qt2 = q.split('#')
            reversing = False
            if pt1 == qt2 or qt1 == pt2:
                # assert qt1 == pt2
                reversing = True
        except Exception as e:
            print('error:', e, l)
        p_temp = pred2template(p)
        q_temp = pred2template(q, reversing=reversing)
        f_write.write(p_temp+'.\t'+q_temp+'.\t'+ans+'\n')
        if n%500 == 0:
            print(n)