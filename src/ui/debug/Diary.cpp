//
// Created by alex2772 on 4/10/26.
//

#include "Diary.h"

#include "DiaryEmbedding.h"
#include "DiaryQueryAI.h"
#include "AUI/View/ATabView.h"

_<AView> ui::debug::Diary::operator()() {
    auto tabs = _new<ATabView>();
    tabs->addTab(DiaryEmbedding{}, "Embedding search");
    tabs->addTab(DiaryQueryAI{}, "queryAI");
    return tabs;
}
