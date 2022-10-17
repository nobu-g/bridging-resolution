import re
from pyknp import BList, Tag
from typing import List, Tuple

from kyoto_reader import BasePhrase, Document

KANJI_PTN = re.compile(r'[\u3005\u3006\u4E00-\u9FFF]+')


def _is_target(tag: Tag) -> bool:
    # 項を取らない体言が対象
    if not ('体言' in tag.features and '非用言格解析' not in tag.features):
        return False

    mrph_list = tag.mrph_list()
    # 名詞などを含まない基本句は除外
    if not any(mrph.hinsi in ('指示詞', '未定義語') or (mrph.hinsi == '名詞' and mrph.bunrui != '形式名詞') for mrph
               in mrph_list):
        return False

    # 数詞を含む基本句は除外
    if any(mrph.bunrui == '数詞' for mrph in mrph_list):
        return False

    # 助詞などが付いている基本句はヘッドなので採用
    if mrph_list[-1].hinsi in ('助詞', '特殊', '判定詞'):
        return True

    if parent := tag.parent:
        # 係り先が次の基本句かつ体言
        if parent.tag_id - tag.tag_id == 1 and '体言' in parent.features:
            if all(mrph.hinsi in ('助詞', '特殊', '判定詞', '接尾辞') for mrph in parent.mrph_list()):
                return True
            if '非用言格解析' in parent.features or '用言' in parent.features:
                return True
            return False
    return True


def is_candidate(bp: BasePhrase, anaphor: BasePhrase, include_intra_sentential_cataphora: bool = False) -> bool:
    # discard self
    if bp.dtid == anaphor.dtid:
        return False

    # discard inter-sentential cataphora
    if bp.dtid > anaphor.dtid:
        if include_intra_sentential_cataphora is False:
            # cataphora
            return False
        if bp.sid != anaphor.sid:
            # inter-sentential cataphora
            return False

    # 体言でないものも捨てる
    if '体言' not in bp.tag.features:
        return False

    # 名詞などを含まない基本句も捨てる
    if not any(mrph.hinsi in ('動詞', '形容詞', '名詞', '指示詞', '未定義語', '接尾辞') for mrph in bp.mrph_list()):
        return False

    return True


def _core(tag: Tag) -> Tuple[str, str, str]:
    """助詞等を除いた中心的表現"""
    mrph_list = tag.mrph_list()
    sidx = 0
    for i, mrph in enumerate(mrph_list):
        if mrph.hinsi not in ('助詞', '特殊', '判定詞', '助動詞', '接続詞'):
            sidx += i
            break
    eidx = len(mrph_list)
    for i, mrph in enumerate(reversed(mrph_list)):
        if mrph.hinsi not in ('助詞', '特殊', '判定詞', '助動詞', '接続詞'):
            eidx -= i
            break
    if sidx >= eidx:
        return '', tag.midasi, ''
    before = ''.join(mrph.midasi for mrph in mrph_list[:sidx])
    content = ''.join(mrph.midasi for mrph in mrph_list[sidx:eidx])
    after = ''.join(mrph.midasi for mrph in mrph_list[eidx:])
    return before, content, after


def _is_kanji(s: str) -> bool:
    return bool(KANJI_PTN.fullmatch(s))


def get_is_compound_subs(blist: BList) -> List[bool]:
    """文中の各基本句が，次に1文字漢字が続く複合語基本句か
    例) "日本 標準 時" -> [False, True, False]
    """
    ret = []
    tag_list = blist.tag_list()
    for tid in range(len(tag_list) - 1):
        tag = tag_list[tid]
        next_tag = tag_list[tid + 1]
        if tag.parent.tag_id == next_tag.tag_id:
            before, content, after = _core(tag)
            if _is_kanji(content) and after == '' and '一文字漢字' in next_tag.features:
                ret.append(True)
                continue
        ret.append(False)
    ret.append(False)
    assert len(tag_list) == len(ret)
    return ret


def get_target_mask(blist: BList) -> List[bool]:
    is_compound_subs = get_is_compound_subs(blist)
    mask: List[bool] = []
    for tag in blist.tag_list():
        mask.append(_is_target(tag) is True and is_compound_subs[tag.tag_id] is False)
    return mask


def get_candidate_mask(anaphor: BasePhrase,
                       document: Document,
                       include_intra_sentential_cataphora: bool = False,
                       ) -> List[bool]:
    mask: List[bool] = []
    for sent in document:
        is_compound_subs = get_is_compound_subs(sent.blist)
        for bp in sent.bps:
            mask.append(
                is_candidate(bp, anaphor, include_intra_sentential_cataphora) and is_compound_subs[bp.tid] is False
            )
    return mask
