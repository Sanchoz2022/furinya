#pragma once

#include "AUI/Common/AString.h"

struct Endpoint {
    AString baseUrl;
    AString bearerKey;
};

struct EndpointAndModel {
    Endpoint endpoint;
    AString model;
};