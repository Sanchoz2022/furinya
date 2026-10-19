#pragma once

#include <telegram/ITelegramClient.h>

namespace util {
AFuture<bool> isAccessibleFromLockdown(ITelegramClient& telegram, int64_t chatId);
}