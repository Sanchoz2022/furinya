//
// Created by alex2772 on 3/14/26.
//

#include "KuniDebugWindow.h"

#include "AUI/Util/Declarative/Containers.h"
#include "AUI/View/ATabView.h"
#include "Diary.h"

KuniDebugWindow::KuniDebugWindow(): AWindow("Kuni: Debug") {
    auto tabs = _new<ATabView>();
    tabs->setExpanding();
    tabs->addTab(ui::debug::Diary{}, "Diary");
    setContents(declarative::Centered::Expanding { tabs });
}
